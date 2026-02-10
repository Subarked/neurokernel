#!/usr/bin/env python

"""
Worker class that operates using mpi
see also: manager.py
"""

import inspect
import os
import re
import subprocess
import sys

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

from neurokernel.mpi_proc import getargnames, Process, ProcessManager
from neurokernel.mixins import LoggerMixin, FakeLoggerMixin
from neurokernel.tools.logging import setup_logger, set_excepthook
from neurokernel.tools.misc import memoized_property, catch_exception

from tqdm import tqdm

class Worker(Process):
    """
    MPI worker class.

    This class repeatedly executes a work method.

    Parameters
    ----------
    ctrl_tag : int
        MPI tag to identify control messages transmitted to worker nodes.
    manager: bool
        Managerless running mode flag. It False, run Module without a
        manager. (default: True).
    """

    def __init__(self, ctrl_tag=1, manager = True, *args, **kwargs):
        super(Worker, self).__init__(manager = manager, *args, **kwargs)

        # Tag used to distinguish control messages:
        self._ctrl_tag = ctrl_tag
        # Execution step counter:
        self.steps = 0
        self.error = False
        self.debug = None
        self.post_run_complete = False # not always set in pre_run

    # Define properties to perform validation when the maximum number of
    # execution steps set:
    _max_steps = float('inf')
    @property
    def max_steps(self):
        """
        Maximum number of steps to execute.
        """
        return self._max_steps
    @max_steps.setter
    def max_steps(self, value):
        if value < 0:
            raise ValueError('invalid maximum number of steps')
        print('maximum number of steps changed: %s -> %s' % \
                      (self._max_steps, value))
        self._max_steps = value

    def do_work(self):
        """
        Work method.

        This method is repeatedly executed by the Worker instance after the
        instance receives a 'start' control message and until it receives a 'stop'
        control message. It should be overridden by child classes.
        """
        pass

    def progressbar_name(self):
        return 'worker'

    def pre_run(self):
        """
        Code to run before main loop.

        This method is invoked by the `run()` method before the main loop is
        started.
        """

        print('running code before body of worker %s' % self.rank)
        self.post_run_complete = False

    def post_run(self):
        """
        Code to run after main loop.

        This method is invoked by the `run()` method after the main loop is
        started.
        """

        self._finalize()

    def _finalize(self):
        if not self.post_run_complete:
            self.pbar.close() # it should've already been closed in `run` but just to make sure.
            print('running code after body of worker %s' % self.rank)
            if self.manager:
                # Send acknowledgment message:
                self.intercomm.send(['done', self.rank], 0, self._ctrl_tag)
                print('done message sent to manager')
            self.post_run_complete = True

    def catch_exception_run(self, func, *args, **kwargs):
        # If the debug flag is set, don't catch exceptions so that
        # errors will lead to visible failures:
        error = catch_exception(func, None, self.debug, *args, **kwargs)
        if self.manager:
            if error is not None:
                if not self.error:
                    self.intercomm.isend(['error', (self.id, self.steps, error)],
                                         dest=0, tag=self._ctrl_tag)
                    print('error sent to manager')
                    self.error = True
        else:
            pass

    def run(self, steps = 0):
        """
        Main body of worker process.
        """

        #self.pre_run()
        self.catch_exception_run(self.pre_run)
        self.pbar = tqdm(desc = self.progressbar_name(), position = self.rank)

        print('running body of worker %s' % self.rank)

        # Start listening for control messages from parent process:
        request = MPI.REQUEST_NULL # REQUEST_NULL gets refreshed in the loop automatically
        running = False
        self.steps = 0
        if not self.manager:
            self.max_steps = steps
            self.pbar.total = self.max_steps
            running = True
        while True:
            # Execute work method; the work method may send data back to the master
            # as a serialized control message containing two elements, e.g.,
            # self.intercomm.isend(['foo', str(self.rank)],
            #                      dest=0, tag=self._ctrl_tag)
            if running:
                self.do_work()
                self.steps += 1
                self.pbar.update()
                #print('execution step: %s' % self.steps)

            # Leave loop if maximum number of steps has been reached:
            if self.steps >= self.max_steps:
                running = False
                print('maximum steps reached')
                break

            if self.manager:
                # Handle control messages (this assumes that only one control
                # message will arrive at a time):
                # refresh our request if its null
                if request == MPI.REQUEST_NULL:
                    request = self.intercomm.irecv(source=0, tag=self._ctrl_tag)
                # check if we have an incoming message, and continue to message logic if yes
                flag, msg_list = request.test()
                if not flag:
                    continue
                # Start executing work method:
                if msg_list[0]== 'start':
                    print('starting')
                    running = True

                # Stop executing work method::
                elif msg_list[0] == 'stop':
                    if self.max_steps == float('inf'):
                        print('stopping')
                        running = False
                    else:
                        print('max steps set - not stopping')
                        pass

                # Set maximum number of execution steps:
                elif msg_list[0] == 'steps':
                    if msg_list[1] == 'inf':
                        self.max_steps = float('inf')
                    else:
                        self.max_steps = int(msg_list[1])
                    self.pbar.total = self.max_steps
                    print('setting maximum steps to %s' % self.max_steps)

                # Quit:
                elif msg_list[0] == 'quit':
                    # if self.max_steps == float('inf'):
                    print('quitting')
                    break
                    # else:
                    #     print('max steps set - not quitting')

        # self.post_run()
        self.catch_exception_run(self.post_run)
        if not self.post_run_complete:
            self._finalize()
