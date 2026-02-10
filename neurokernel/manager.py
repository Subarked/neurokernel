#!/usr/bin/env python

"""
Manager class that handles execution of several workers
see also: worker.py
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
from neurokernel.worker import Worker

from tqdm import tqdm

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
