
from plm_base import *
from dataset import *
from utils import *
from embedder import *
from example_run_jobs import *

import pandas as pd
import argparse
import os
import datetime
import json
import re

CONDA_ENV_NAME = "esm_env"
KEY_DATASET_LOG_PATH = "all_keydatasets_log.csv"
CHUNK_SIZE = 500

def generate_random_hash(length=32):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def ab_run_dataset(model, dataset_path, indices=None, heavy_or_light_only=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print(f"\t\t[INFO] Using device: {device}")

    ab_model = None

    if indices != -1:
        indices_str = indices
        indices = indices.split("_")
        indices = [i for i in range(int(indices[0]), int(indices[1]))]
        eval_path = dataset_path + f"/{indices_str}"
        os.makedirs(eval_path, exist_ok=True)
    else:
        indices = None
        eval_path = dataset_path

    dataset_path = dataset_path + "/sequences.csv"

    train_test_dataset = PREActivityDataset(
        train_project_name="",
        encoding_identifier="",
        evaluation_path=eval_path,
        dataset_path=dataset_path,
        train_indices=indices,
        test_indices=None,
        encoding_function=model.encode,
        cache=True,
        lazy_load=True,
        sequence_column_name="formatted_sequences",
        label_column_name="index",
        labels_dtype=torch.float32,
        device=device
    )

    train_test_dataset.evaluate(
        model,
        ab_embedding_evaluate_heavy_or_light_chain if heavy_or_light_only is not None else ab_embeddings_evaluate_function, 
        ab_embeddings_finalize_function,
        eval_train=True,
        eval_test=False,
        internal_batch_size=20
    )

def run_dataset(model, dataset_path, indices=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print(f"\t\t[INFO] Using device: {device}")

    ab_model = None

    if indices != -1:
        indices_str = indices
        indices = indices.split("_")
        indices = [i for i in range(int(indices[0]), int(indices[1]))]
        eval_path = dataset_path + f"/{indices_str}"
        os.makedirs(eval_path, exist_ok=True)
    else:
        indices = None
        eval_path = dataset_path

    dataset_path = dataset_path + "/sequences.csv"

    train_test_dataset = PREActivityDataset(
        train_project_name="",
        encoding_identifier="",
        evaluation_path=eval_path,
        dataset_path=dataset_path,
        train_indices=indices,
        test_indices=None,
        encoding_function=model.encode,
        cache=True,
        lazy_load=True,
        sequence_column_name="formatted_sequences",
        label_column_name="index",
        labels_dtype=torch.float32,
        device=device
    )

    train_test_dataset.evaluate(
        model,
        ab_embedding_evaluate_heavy_or_light_chain if heavy_or_light_only is not None else ab_embeddings_evaluate_function, 
        ab_embeddings_finalize_function,
        eval_train=True,
        eval_test=False,
        internal_batch_size=20
    )


def _lock_wrapper(file_name):
    pass

def _unlock_wrapper(file_name):  
    pass

# key_dataset_paths = ["/home/labs/fleishman/itayta/CADABRE/cadabre_works_key_dataset.csv",
#                      "/home/labs/fleishman/itayta/CADABRE/jobs_key_dataset.csv"]

# #for key_dataset_path in key_dataset_paths:
# worker(key_dataset_paths[0])


def __validate_dataset(args):
    if not hasattr(args, 'dataset_file') or args.dataset_file is None:
        raise ValueError("You must provide a dataset_file argument for embed_parallel/worker.")

    dataset_file = args.dataset_file

    if not dataset_file.endswith('.csv'):
        raise ValueError(f"Provided dataset_file '{dataset_file}' is not a CSV file.")

    sequence_colname = getattr(args, "sequence_colname", "full_sequence")

    try:
        df = pd.read_csv(dataset_file)
    except Exception as e:
        raise ValueError(f"Could not read csv '{dataset_file}': {e}")

    if sequence_colname not in df.columns:
        raise ValueError(f"Column '{sequence_colname}' not found in '{dataset_file}'. Please provide the correct --sequence_colname argument.")

    print(f"[INFO] Successfully loaded '{dataset_file}' with column '{sequence_colname}'.")

    return df

def worker(args):
    if not hasattr(args, "jobs_path") or args.jobs_path is None:
        raise ValueError("--jobs_path must be provided as an argument.")

    jobs_args_path = os.path.join(args.jobs_path, "job_args.json")

    if not os.path.exists(jobs_args_path):
        raise FileNotFoundError(f"jobs_args.json not found in jobs_path: {jobs_args_path}")

    print(f"[INFO] Reading job arguments from: {jobs_args_path}")

    with open(jobs_args_path, "r") as f:
        job_args_dict = json.load(f)

    job_args = argparse.Namespace(**job_args_dict)
    _ = __validate_dataset(job_args)

    key_dataset_path = os.path.join(args.jobs_path, "key_dataset.csv")

    if not os.path.exists(key_dataset_path):
        raise FileNotFoundError(f"key_dataset.csv not found at: {key_dataset_path}")
    
    is_working = True

    while is_working:

        torch.cuda.empty_cache()
        sleep_time = random.uniform(1, 20)
        print(f"[INFO] Sleeping for {sleep_time:.2f} seconds before checking for jobs.")
        time.sleep(sleep_time)
        

        job_key_dataset = pd.read_csv(key_dataset_path)
        jobs = job_key_dataset["job_path"]

        job_found = False
        for job_file_path in jobs:
            # Check if job file exists
            if not os.path.exists(job_file_path):
                print(f"[INFO] Job {job_file_path} does not exist, skipping.")
                continue

            try:
                # Try to open and lock the file exclusively
                with open(job_file_path, "r+") as job_f:
                    try:
                        fcntl.flock(job_f, fcntl.LOCK_EX | fcntl.LOCK_NB)  # non-blocking lock
                    except Exception as e:
                        # File is locked by another worker, skip to next file
                        print(f"[INFO] Job {job_file_path} is locked by another worker, skipping.")
                        print(f"[ERROR] Error: {e}")
                        continue

                    completed_successfully = False

                    try:
                        # Read the job file, perform the actual work here
                        job_df = pd.read_csv(job_f)
                        print(f"[INFO] Processing job: {job_file_path}")

                        
                        dataset_path = job_key_dataset[job_key_dataset["job_path"] == job_file_path]["ds_path"].values[0]
                        indices = job_key_dataset[job_key_dataset["job_path"] == job_file_path]["indices"].values[0]

                        if indices and isinstance(indices, str):
                            m = re.match(r'^(\d+)_(\d+)$', indices)
                            if m:
                                indices = list(range(int(m.group(1)), int(m.group(2)) + 1))
                        elif not indices or str(indices).lower() == 'none':
                            indices = None
                        ab_run_dataset(dataset_path, indices)
                        completed_successfully = True

                        # After successful processing, close and delete the job file
                    except Exception as e:
                        print(f"[ERROR] Failed to process {job_file_path}: {e}")
                        # Release the lock before continuing
                        fcntl.flock(job_f, fcntl.LOCK_UN)
                        continue

                    finally:
                        # Always unlock before closing
                        job_found = True
                        if completed_successfully:
                            os.remove(job_file_path)
                            print(f"[INFO] Successfully completed and deleted {job_file_path}")
                        fcntl.flock(job_f, fcntl.LOCK_UN)

                        
            except Exception as e:
                print(f"[ERROR] Could not process {job_file_path}: {e}")
                continue  # Process only one job per loop iteration

        if not job_found:
            # No available jobs to process, worker can exit or sleep + loop
            print("[INFO] No jobs left to process. Worker exiting.")
            is_working = False


def list_jobs(args):
    if not os.path.exists(KEY_DATASET_LOG_PATH):
        print("No jobs found. (keydataset_log_path does not exist)")
        return

    try:
        df = pd.read_csv(KEY_DATASET_LOG_PATH)
    except Exception as e:
        print(f"[ERROR] Could not read {KEY_DATASET_LOG_PATH}: {e}")
        print("No jobs found.")
        return

    if df.empty:
        print("No jobs found. (log file is empty)")
        return

    for idx, row in df.iterrows():
        job_path = row.get("jobs_path", "<no job_path>")
        n_jobs = row.get("num_jobs", "<0>")
        dataset_name = row.get("dataset_name", "<no dataset_name>")
        print(f"{idx}. Job: {job_path} \nNumber of jobs: {n_jobs} \nDataset name: {dataset_name}\n")

def embed_sequences(args):
    pass

def embed_dataset(args):
    pass

def embed_parallel(args):        
    df = __validate_dataset(args)

    # Check if evaluation_path is provided as an argument; if not, raise error
    eval_path = getattr(args, "evaluation_path", None)

    if eval_path is None:

        raise ValueError("You must provide an evaluation_path argument for embed_parallel.")

    os.makedirs(eval_path, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs_path = eval_path + f"/jobs_{timestamp}_parallel"

    os.makedirs(jobs_path, exist_ok=True)
    os.makedirs(os.path.join(jobs_path, "jobs"), exist_ok=True)

    print(f"[INFO] Using evaluation_path: {eval_path}")

    model = plmEmbeddingModel(plm_name=args.model)
    
    # Just to cache
    train_test_dataset = PREActivityDataset(
        train_project_name="",
        encoding_identifier=args.model,
        evaluation_path=eval_path,
        dataset_path=args.dataset_file,
        train_indices=None,
        test_indices=None,
        encoding_function=model.encode,
        cache=True,
        lazy_load=True,
        sequence_column_name=args.sequence_colname,
        label_column_name=None,
        labels_dtype=torch.float32)
        
    N_sequences = len(df)

    print(f"[INFO] Creating jobs in: {jobs_path}")
    num_chunks = N_sequences // CHUNK_SIZE

    key_dataset = []

    for i in range(num_chunks + 1):
        start_idx = i * CHUNK_SIZE
        end_idx = start_idx + CHUNK_SIZE
        if end_idx > N_sequences:
            end_idx = N_sequences
        indices = f"{start_idx}_{end_idx}"
        print(indices)
        # Generate job_path as in example code
        chunk_job_path = os.path.join(jobs_path, "jobs", generate_random_hash() + ".job")

        with open(chunk_job_path, "w") as f:
            pass

        key_dataset.append({
            "ds_name": args.dataset_file,
            "ds_path": args.dataset_file,
            "job_path": chunk_job_path,
            "indices": indices,
            "N": (end_idx - start_idx)
        })

    key_dataset_df = pd.DataFrame(key_dataset)
    key_dataset_save_path = jobs_path + "/key_dataset.csv"
    key_dataset_df.to_csv(key_dataset_save_path, index=False)

    print(f"[INFO] Created {len(key_dataset)} jobs in: {jobs_path}")

        
    keydataset_log_path = KEY_DATASET_LOG_PATH

    log_columns = ["dataset_name", "num_jobs", "jobs_path", "eval_path"]

    # Try to read existing log, otherwise create
    if os.path.exists(keydataset_log_path):
        keydataset_log_df = pd.read_csv(keydataset_log_path)
    else:
        keydataset_log_df = pd.DataFrame(columns=log_columns)

    log_entry = {
        "dataset_name": os.path.basename(args.dataset_file),
        "num_jobs": len(key_dataset),
        "jobs_path": jobs_path,
        "eval_path": eval_path,
    }

    keydataset_log_df = pd.concat([keydataset_log_df, pd.DataFrame([log_entry])], ignore_index=True)
    keydataset_log_df.to_csv(keydataset_log_path, index=False)

    print(f"[INFO] Registered job for dataset {os.path.basename(args.dataset_file)} with {len(key_dataset)} chunks in log: {keydataset_log_path}")
    
    args_dict = vars(args).copy()
    args_dict = dict([(k,v) for k,v in args_dict.items() if v is not None])
    args_dict.pop("action", None)
    args_args_path = os.path.join(jobs_path, "job_args.json")

    args.jobs_path = jobs_path

    with open(args_args_path, "w") as f:
        json.dump(args_dict, f)

    execute_workers(args)

def embed_view_job_file(args):
    pass

def execute_workers(args):
    if not hasattr(args, "jobs_path") or args.jobs_path is None:
        raise ValueError("jobs_path is required in args for execute_workers.")

    jobs_path = args.jobs_path

    if not os.path.exists(jobs_path):
        raise FileNotFoundError(f"jobs_path does not exist: {jobs_path}")
    
    cmd = f"python {os.path.basename(__file__)} --action worker --jobs_path {jobs_path}"

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    job_exec_lib_name = f"job_execution_{timestamp}"
    job_exec_lib_path = os.path.join(jobs_path, job_exec_lib_name)
    os.makedirs(job_exec_lib_path, exist_ok=True)


    print(f"[INFO] Created job execution library at: {job_exec_lib_path}")

    workers_metadata = []
    for i in range(args.n_workers):
        if args.job_executing_type == "lsf":
            job_metadata = \
                run_lsf_job(cmd, 
                            job_exec_lib_path,
                            os.getcwd(), 
                            CONDA_ENV_NAME,
                            worker_idx=i)

            workers_metadata.append({"script_path": job_metadata[0], 
                                     "err_file": job_metadata[1], 
                                     "out_file": job_metadata[2],
                                     "worker_idx": i})

        elif args.job_executing_type == "local":
            # ToDo: add local job execution
            raise NotImplementedError("Local job execution is not implemented yet.")
        else:
            raise ValueError(f"Unknown job_executing_type: {args.job_executing_type}")

    workers_metadata_csv_path = os.path.join(job_exec_lib_path, "workers_metadata.csv")
    df = pd.DataFrame(workers_metadata)
    df.to_csv(workers_metadata_csv_path, index=False)
    
    print(f"[INFO] Wrote workers metadata to: {workers_metadata_csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--average_embeddings', action='store_true', default=False)
    parser.add_argument('--positions_to_embed', type=int, nargs='+')
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--n_workers', default=5,type=int)
    parser.add_argument('--sequences', nargs='+')
    parser.add_argument('--model', type=str, default="esm2_t12_35M_UR50D")
    parser.add_argument('--pretrained_weights', type=str)
    parser.add_argument('--outputfile', type=str)
    parser.add_argument('--evaluation_path', type=str, help="Path to the evaluation output or evaluation-related files.")
    parser.add_argument('--jobs_path', type=str, help="Path to the directory of the jobs.")
    parser.add_argument('--sequence_colname', type=str, default="full_sequence", help="Column name in the CSV file containing the sequences.")
    parser.add_argument('--job_executing_type', type=str, default='local', choices=['lsf', 'local'], help="Type of job execution: 'lsf' (submit jobs via LSF cluster) or 'local' (run as local threads).")
    parser.add_argument('--action', type=str, required=True, choices=['embed_sequences', 'embed_dataset', 'embed_parallel', 'embed_view_job_file', 'list_jobs'])
    args = parser.parse_args()

    if args.action == 'embed_sequences':
        embed_sequences(args)
    elif args.action == 'embed_dataset':
        embed_dataset(args)
    elif args.action == 'embed_parallel':
        embed_parallel(args)
    elif args.action == 'embed_view_job_file':
        embed_view_job_file(args)
    elif args.action == 'list_jobs':
        list_jobs(args)
    elif args.action == 'worker':
        worker(args)
    else:
        pass

if __name__ == "__main__":
    fitness_learning_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning"

    # Add the root of the library *and* its internal 'code' folder to sys.path for intra-library imports
    sys.path.append(fitness_learning_path)
    sys.path.append(os.path.join(fitness_learning_path, "code"))

    plm_init(fitness_learning_path)

    #ablang_model = abPlmEmbeddingModel(plm_name="igbert")
    #esm2_model = plmEmbeddingModel(plm_name="esm2_t12_35M_UR50D")

    #ab_run_dataset(ablang_model, "/home/labs/fleishman/itayta/CADABRE/OAS_datasets/OAS_10000_10000.csv", "10000_10000")

    main()