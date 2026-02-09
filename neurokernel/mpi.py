#!/usr/bin/env python

"""
MPI support classes.
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


class WorkerManager(ProcessManager):
    """
    Self-launching MPI worker manager.

    This class may be used to construct an MPI application consisting of

    - a manager process that spawns MPI processes that execute the run() methods
      of several subclasses of the Worker class;
    - worker processes that perform some processing task; and

    The application should NOT be started via mpiexec.

    Parameters
    ----------
    ctrl_tag : int
        MPI tag to identify control messages transmitted to worker nodes.
        May not be equal to mpi4py.MPI.ANY_TAG

    Notes
    -----
    This class requires MPI-2 dynamic processing management.

    See Also
    --------
    Worker
    """

    def __init__(self, ctrl_tag=1):
        super(WorkerManager, self).__init__()

        # Validate control tag.
        assert ctrl_tag != MPI.ANY_TAG

        # Tag used to distinguish MPI control messages:
        self._ctrl_tag = ctrl_tag

    def add(self, target, *args, **kwargs):
        """
        Add a worker to an MPI application.

        Parameters
        ----------
        target : Worker
            Worker class to instantiate and run.
        args : sequence
            Sequential arguments to pass to target class constructor.
        kwargs : dict
            Named arguments to pass to target class constructor.
        """

        assert issubclass(target, Worker)
        print('adding class %s' % target.__name__)
        return ProcessManager.add(self, target, *args, **kwargs)

    def process_worker_msg(self, msg):
        """
        Process the specified deserialized message from a worker.
        """
        print('got ctrl msg: %s' % str(msg))

    def wait(self):
        """
        Wait for execution to complete.
        """
        # Start listening for control messages
        active_workers = list(range(len(self))) # keep a list of active workers in case a worker re-sends a done message
        req = MPI.Request()
        while True:
            # Check for control messages from workers:
            msg_list = self.intercomm.recv(source=MPI.ANY_SOURCE, tag=self._ctrl_tag)
            print('MANAGER received message \'%s\'' % msg_list[0])
            if msg_list[0] == 'done':
                print('removing %s from worker list' % msg_list[1])
                if not msg_list[1] in active_workers:
                    print('worker %s sent a duplicate \'done\' request. This is a bug!' % msg_list[1])
                else:
                    active_workers.remove(msg_list[1])

            # Additional control messages from the workers are processed
            # here:
            else:
                self.process_worker_msg(msg_list[0])

            if not active_workers:
                print('finished running manager')
                break

    def start(self, steps=float('inf')):
        """
        Tell the workers to start processing data.
        """
        print('MANAGER start')
        print('sending steps message (%s)' % steps)
        for dest in range(len(self)):
            print('MANAGER --steps--> %d' % dest)
            self.intercomm.isend(['steps', str(steps)], dest, self._ctrl_tag)
        print('sending start message')
        for dest in range(len(self)):
            print('MANAGER --start--> %d' % dest)
            self.intercomm.isend(['start'], dest, self._ctrl_tag)

    def stop(self):
        """
        Tell the workers to stop processing data.
        """
        print('MANAGER stop')
        print('sending stop message')
        for dest in range(len(self)):
            print('MANAGER --stop--> %d' % dest)
            self.intercomm.isend(['stop'], dest, self._ctrl_tag)

    def quit(self):
        """
        Tell the workers to quit.
        """
        print('MANAGER quit')
        print('sending quit message')
        for dest in range(len(self)):
            print('MANAGER --quit--> %d' % dest)
            self.intercomm.isend(['quit'], dest, self._ctrl_tag)

if __name__ == '__main__':
    import neurokernel.mpi_relaunch
    import time
    MPI.Init()
    #setup_logger(screen=True, file_name='neurokernel.log',
    #        mpi_comm=MPI.COMM_WORLD, multiline=True)

    # Define a class whose constructor takes arguments so as to test
    # instantiation of the class by the manager:
    class MyWorker(Worker):
        def __init__(self, x, y, z=None, routing_table=None):
            super(MyWorker, self).__init__()
            name = MPI.Get_processor_name()
            print('I am process %d of %d on %s.' % (self.rank,
                                                           self.size, name))
            print('init args: %s, %s, %s' % (x, y, z))

    man = WorkerManager()
    man.add(target=MyWorker, x=1, y=2, z=3)
    man.add(MyWorker, 3, 4, 5)
    man.add(MyWorker, 6, 7, 8)
    man.spawn()
    # To run for a specific number of steps, run
    # man.start(number_of_steps)
    man.start(100)
    man.wait()
