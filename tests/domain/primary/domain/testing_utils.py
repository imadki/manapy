from mpi4py import MPI
import time

def log_step(string = ""):
  if not hasattr(log_step, "start"):
    def print_results():
      dic = log_step.dic
      sorted_dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
      print("------------------------------------------------")
      print(">>>>>>>>>>>>>>>>>> Results <<<<<<<<<<<<<<<<<<<<<")
      print("------------------------------------------------")
      for item in sorted_dic:
        print(f'{item} => {sorted_dic[item]:.6f} seconds')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    log_step.start = time.time()
    log_step.old_time = time.time()
    log_step.rank = rank
    log_step.dic = {}
    log_step.print_resutls = print_results
    log_step.file = f'results_{rank}.log'
  if string != "":
    name = f"[Rank {log_step.rank}]: {string}"
    print(name, end='')
    log_step.step_name = name
  else:
    time_taken = time.time()-log_step.old_time
    print(f" Time {time.time()-log_step.start:.6f} seconds (delta: {time_taken:.6f} seconds)", flush=True)
    log_step.old_time = time.time()
    log_step.dic[log_step.step_name] = time_taken
