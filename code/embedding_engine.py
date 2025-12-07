
from plm_base import *
from dataset import *
from utils import *
from embedder import *


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
        activity_column_name="index",
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

def worker(key_dataset_path):
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

    

key_dataset_paths = ["/home/labs/fleishman/itayta/CADABRE/cadabre_works_key_dataset.csv",
                     "/home/labs/fleishman/itayta/CADABRE/jobs_key_dataset.csv"]

#for key_dataset_path in key_dataset_paths:
worker(key_dataset_paths[0])


if __name__ == "__main__":
    fitness_learning_path = "/home/labs/fleishman/itayta/fitness/"

    # Add the root of the library *and* its internal 'code' folder to sys.path for intra-library imports
    sys.path.append(fitness_learning_path)
    sys.path.append(os.path.join(fitness_learning_path, "code"))

    plm_init(fitness_learning_path)

    ablang_model = abPlmEmbeddingModel(plm_name="igbert")
    esm2_model = plmEmbeddingModel(plm_name="esm2_t12_35M_UR50D")

    ab_run_dataset(ablang_model, "/home/labs/fleishman/itayta/CADABRE/OAS_datasets/OAS_10000_10000.csv", "10000_10000")