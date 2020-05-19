from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def _train_faces_iteration(n):
    return n

pool = Pool(cpu_count())
faces = list(
        tqdm(
            pool.imap(_train_faces_iteration, [1, 2, 3, 4, 5, 6, 7, 8, 9]), 
            total=9))
pool.close()
pool.join()

print(faces)


