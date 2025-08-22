import os
from mpi4py import MPI
import jittor as jt
import numpy as np
import time
import threading

class DistributedManager:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.comm_times = {
            'broadcast': 0.0
        }

    def reset_comm_times(self):
        """ 重置计时 """
        for key in self.comm_times:
            self.comm_times[key] = 0.0

    def report_comm_times(self):
        """ 打印当前进程的通信时间统计 """
        print(f"--- Rank {self.rank} Communication Times (seconds) ---")
        for op, t in self.comm_times.items():
            print(f"{op:<45}: {t:.6f}")
        print("-------------------------------------------------")

    def broadcast_args(self, args):
        return self.comm.bcast(args, root=0)

    def synchronize_model_parameters(self, model):
        for param in model.parameters():
            param_np = param.numpy()
            self.comm.Allreduce(MPI.IN_PLACE, param_np, op=MPI.SUM)
            param.update(jt.array(param_np / self.size))

    def synchronize_model_parameters_non_blocking(self, model):
        """
        非阻塞参数同步
        举例：
        # 非阻塞参数同步
        requests = manager.synchronize_model_parameters_non_blocking(model)
        # 在通信的同时可以执行其他任务（例如打印日志）
        print(f"Rank {manager.rank}: Performing other tasks while waiting for synchronization...")
        # 完成通信并更新参数
        manager.finalize_synchronization(requests)
        """
        requests = []
        for param in model.parameters():
            param_np = param.numpy()
            request = self.comm.Iallreduce(MPI.IN_PLACE, param_np, op=MPI.SUM)
            requests.append((param, param_np, request))
        return requests

    def finalize_synchronization(self, requests):
        for param, param_np, request in requests:
            request.Wait()
            param.update(jt.array(param_np / self.size))
            # 这里不一定非得平均，可以加权平均，可以累积，可以自定义聚合

    def finalize(self):
        MPI.Finalize()

    def broadcast(self, data : jt.var, src = 0):
        data_np = np.array(data)
        
        start_time = time.perf_counter()
        result = self.comm.bcast(data_np, root = src)
        end_time = time.perf_counter()
        self.comm_times['broadcast'] += (end_time - start_time)

        return jt.array(result)
    

def async_task(func, *args):
        # 异步执行
        # 也就是你可以做某件事途中突然指定某个函数，异步执行它
        """
        # 异步执行其他任务
        thread = async_task(lambda: print(f"Rank {manager.rank}: Performing other tasks..."))
        # 等待它任务完成
        thread.join()
        # 可以夹在那个非阻塞参数同步中间
        """
        thread = threading.Thread(target=func, args=args)
        thread.start()
        return thread
