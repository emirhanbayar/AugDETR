# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import sys
import uuid
from pathlib import Path
import subprocess
import main as detection
import submitit

def parse_args():
    detection_parser = detection.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for detection", parents=[detection_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=60, type=int, help="Duration of the job")
    parser.add_argument("--cpus_per_task", default=16, type=int, help="CPUs per task")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--job_name", type=str, help="Job name.")
    parser.add_argument("--qos", type=str, default=None, help="specify preemptive QOS.")
    parser.add_argument("--requeue", action='store_true', help="job requeue if preempted.")
    parser.add_argument("--mail_type", type=str, default='ALL', help="send email when job begins, ends, fails or preempted.")
    parser.add_argument("--mail_user", type=str, default='', help="email address.")
    return parser.parse_args()

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup_environment(self):
        """Setup CUDA and other environment variables on the compute node"""
        # CUDA setup
        cuda_path = "/usr/local/cuda"
        os.environ["CUDA_HOME"] = cuda_path
        os.environ["CUDA_PATH"] = cuda_path
        os.environ["PATH"] = f"{cuda_path}/bin:{os.environ['PATH']}"
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
        
        # Fix MKL threading layer conflict
        os.environ["MKL_THREADING_LAYER"] = "GNU"
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
        
        return cuda_path
    
    def verify_cuda_setup(self):
        """Verify CUDA installation and torch.cuda availability"""
        import torch
        print("\nCUDA Environment Variables:")
        print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
        print(f"CUDA_PATH: {os.environ.get('CUDA_PATH')}")
        print(f"PATH: {os.environ.get('PATH')}")
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
        
        print("\nPyTorch CUDA Status:")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
            print(f"torch.version.cuda: {torch.version.cuda}")
        
        try:
            nvcc_output = subprocess.check_output(['nvcc', '--version']).decode()
            print("\nnvcc version:")
            print(nvcc_output)
        except Exception as e:
            print(f"Error checking nvcc: {e}")

    def compile_cuda_extensions(self, cuda_path):
        """Compile CUDA extensions including MultiScaleDeformableAttn"""
        current_dir = os.getcwd()
        extension_dir = os.path.join(current_dir, "models/ops")
        
        if not os.path.exists(extension_dir):
            extension_dir = os.path.join(current_dir, "models/dino/ops")
        
        if os.path.exists(extension_dir):
            print(f"Compiling CUDA extensions in {extension_dir}")
            env = os.environ.copy()
            
            # Verify CUDA setup before compilation
            self.verify_cuda_setup()
            
            try:
                # Clean any previous builds
                subprocess.run(
                    ["rm", "-rf", "build"],
                    cwd=extension_dir,
                    check=True
                )
                
                # Run compilation with detailed output
                process = subprocess.Popen(
                    ["/home/ebayar/anaconda3/envs/dino/bin/python", "setup.py", "build", "-v", "install"],
                    cwd=extension_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                stdout, stderr = process.communicate()
                print("\nCompilation output:")
                print(stdout)
                print("\nCompilation errors:")
                print(stderr)
                
                if process.returncode != 0:
                    print("\nCompilation failed! Setup.py contents:")
                    with open(os.path.join(extension_dir, "setup.py"), 'r') as f:
                        print(f.read())
                    raise subprocess.CalledProcessError(process.returncode, process.args)
                    
                print("Successfully compiled CUDA extensions")
                
            except subprocess.CalledProcessError as e:
                print(f"Failed to compile CUDA extensions: {e}")
                raise
        else:
            print(f"Warning: Could not find extension directory at {extension_dir}")

    def __call__(self):
        try:
            # Setup environment on compute node
            cuda_path = self.setup_environment()
            
            # Import numpy after setting MKL variables to avoid threading issues
            import numpy
            
            # Compile CUDA extensions
            # self.compile_cuda_extensions(cuda_path)
            
            # Setup GPU args
            self._setup_gpu_args()
            
            # Run main training
            detection.main(self.args)
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def checkpoint(self):
        import os
        import submitit

        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        job_env = submitit.JobEnvironment()
        self.args.output_dir = self.args.job_dir
        self.args.output_dir = str(self.args.output_dir).replace("%j", str(job_env.job_id))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

def get_init_file():
    shared_folder = Path(os.getenv("HOME")) / "experiments"
    shared_folder.mkdir(exist_ok=True)
    init_file = shared_folder / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

def main():
    args = parse_args()
    args.command_txt = "Command: " + ' '.join(sys.argv)
    
    if args.job_dir == "":
        raise ValueError("You must set job_dir manually.")

    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    additional_parameters = {
        'mail-user': args.mail_user,
        'mail-type': args.mail_type,
    }
    if args.requeue:
        additional_parameters['requeue'] = args.requeue

    executor.update_parameters(
        mem_gb=0,
        gpus_per_node=args.ngpus,
        tasks_per_node=args.ngpus,  # one task per GPU
        cpus_per_task=args.cpus_per_task,
        nodes=args.nodes,
        timeout_min=args.timeout,
        qos=args.qos,
        slurm_additional_parameters=additional_parameters
    )

    executor.update_parameters(name=args.job_name)
    args.dist_url = get_init_file().as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)
    print("Submitted job_id:", job.job_id)

if __name__ == "__main__":
    main()