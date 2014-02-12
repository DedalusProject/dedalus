# Copyright (c) 2013-2014, Keaton J. Burns.
#
# This file is part of Dedalus, which is free software distributed
# under the terms of the GPLv3 license.  A copy of the license should
# have been included in the file 'LICENSE.txt', and is also available
# online at <http://www.gnu.org/licenses/gpl-3.0.html>.

"""
Class for centralized evaluation of expression trees.

"""

import h5py
from mpi4py import MPI
import numpy as np
from .system import FieldSystem

from ..tools.general import OrderedSet


class Evaluator:

    def __init__(self, domain, vars):

        self.domain = domain
        self.vars = vars
        self.handlers = []

    def add_system_handler(self, **kw):

        SH = SystemHandler(self.domain, self.vars, **kw)
        self.handlers.append(SH)

        return SH

    def add_file_handler(self, filename, **kw):

        FH = FileHandler(filename, self.vars, **kw)
        self.handlers.append(FH)

        return FH

    # def add_snapshot_handler(self, filename):

    #     SH = SystemHandler(system, self.vars)
    #     self.handlers.append(SH)

    #     return SH

    def evaluate(self, wall_time, sim_time, iteration, force=False):

        # Find scheduled tasks
        current_handlers = []
        tasks = []
        for handler in self.handlers:

            wall_div = wall_time // handler.wall_dt
            sim_div  = sim_time  // handler.sim_dt
            iter_div = iteration // handler.iter

            wall_up = (wall_div > handler.wall_div)
            sim_up  = (sim_div  > handler.sim_div)
            iter_up = (iter_div > handler.iter_div)

            if force or any((wall_up, sim_up, iter_up)):
                current_handlers.append(handler)
                tasks.extend(handler.tasks)
                handler.wall_div = wall_div
                handler.sim_div  = sim_div
                handler.iter_div = iter_div

        # Start from coefficient space
        fields = self.get_fields(tasks)
        for f in fields:
            f.require_coeff_space()
        L = 0
        tasks = self.attempt_tasks(tasks)

        # Oscillate through layouts until all tasks are evaluated
        Lmax = self.domain.distributor.grid_layout.index
        while tasks:
            # Change direction at first and last layouts
            if L == 0:
                dL = 1
            elif L == Lmax:
                dL = -1
            # Transform fields
            fields = self.get_fields(tasks)
            for f in fields:
                if dL > 0:
                    f.towards_grid_space()
                else:
                    f.toward_coeff_space()
            L += dL
            # Attempt evaluation
            tasks = self.attempt_tasks(tasks)

        # Process
        for handler in current_handlers:
            handler.process(wall_time, sim_time, iteration)


    @staticmethod
    def get_fields(tasks):

        fields = OrderedSet()
        for task in tasks:
            if not np.isscalar(task['operator']):
                fields.update(task['operator'].field_set())

        return fields

    @staticmethod
    def attempt_tasks(tasks):

        unfinished = []
        for task in tasks:
            if np.isscalar(task['operator']):
                task['out']['g'] = task['operator']
            else:
                output = task['operator'].attempt()
                if output is None:
                    unfinished.append(task)
                else:
                    task['out'] = output
                    # layout = output.layout
                    # task['out'][layout] = output.data

        return unfinished


class Handler:

    def __init__(self, vars, wall_dt=np.inf, sim_dt=np.inf, iter=np.inf):

        self.vars = vars
        self.tasks = []

        self.wall_dt = wall_dt
        self.sim_dt = sim_dt
        self.iter = iter

        self.wall_div = 0.
        self.sim_div = 0.
        self.iter_div = 0.

    def add_task(self, task_str, layout='g', name=None):

        # Default name
        if name is None:
            name = task_str

        # Build task dictionary
        task = dict()
        task['operator'] = eval(task_str, self.vars)
        task['layout'] = layout
        task['name'] = name

        self.tasks.append(task)

class SystemHandler(Handler):

    def __init__(self, domain, *args, **kw):

        self.domain = domain
        Handler.__init__(self, *args, **kw)

    def add_tasks(self, tasks):

        for task in tasks:
            self.add_task(task)

    def build_system(self):

        nfields = len(self.tasks)
        self.system = FieldSystem(range(nfields), self.domain)

        for i, task in enumerate(self.tasks):
            if not np.isscalar(task['operator']):
                task['operator'].out = self.system.fields[i]
            else:
                task['out'] = self.system.fields[i]

        return self.system

    def process(self, wall_tiem, sim_time, iteration):

        self.system.gather()


class FileHandler(Handler):

    def __init__(self, filename, *args, **kw):

        self.file_base = filename
        self.file_num = 1
        Handler.__init__(self, *args, **kw)

    def process(self, wall_time, sim_time, iteration):

        # Build filename
        filename = self.file_base + '%06i.hdf5' %self.file_num

        # Create HDF5 structure
        file = h5py.File(filename, 'w', driver='mpio', comm=MPI.COMM_WORLD)
        #scale_group = file.create_group('scales')
        task_group = file.create_group('tasks')

        file.attrs['wall_time'] = wall_time
        file.attrs['sim_time'] = sim_time
        file.attrs['iteration'] = iteration

        # Create task datasets
        for task in self.tasks:
            out = task['out']
            out.require_layout(task['layout'])
            name = task['name']
            shape = out.layout.global_shape
            dtype = out.layout.dtype

            dset = task_group.create_dataset(name=name, shape=shape, dtype=dtype)
            dset[out.layout.slices] = out.data

        file.close()
        self.file_num += 1

#     def add_task(self, )
