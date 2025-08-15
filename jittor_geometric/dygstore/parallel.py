from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

class ParallelProcessor:
    def __init__(self, max_analyzer_threads=1, max_processor_threads=1, max_event_workers=4, queue_maxsize=5, use_processes=False):
        """
        Initialize the parallel processor.

        Args:
            max_analyzer_threads (int): Number of threads/processes for analyzer tasks.
            max_processor_threads (int): Number of threads/processes for processor tasks.
            max_event_workers (int): Maximum workers for event-level parallelism.
            queue_maxsize (int): Maximum size of the queue between analyzer and processor.
            use_processes (bool): Use ProcessPoolExecutor instead of ThreadPoolExecutor for event processing.
        """
        self.max_analyzer_threads = max_analyzer_threads
        self.max_processor_threads = max_processor_threads
        self.max_event_workers = max_event_workers
        self.queue_maxsize = queue_maxsize
        self.use_processes = use_processes
        self.analyzer_to_processor_queue = Queue(maxsize=queue_maxsize)
        self.process_results_queue = Queue()

    def _run_analyzer(self, analyzer_func, chunks, dyrep_obj):
        """
        Run the analyzer function and push results to the queue.
        """
        try:
            analyzer_func(chunks, dyrep_obj, self.analyzer_to_processor_queue)
        finally:
            # Signal end of analysis
            self.analyzer_to_processor_queue.put(None)

    def _run_processor(self, processor_func, dyrep_obj, chunks):
        """
        Run the processor function and push results to the results queue.
        """
        processor_func(dyrep_obj, self.analyzer_to_processor_queue, self.process_results_queue, chunks)

    def execute_pipeline(self, analyzer_func, processor_func, dyrep_obj, chunks):
        """
        Execute the analyzer and processor in a pipelined manner.

        Args:
            analyzer_func (callable): Function to analyze chunks and produce batches.
            processor_func (callable): Function to process batches.
            dyrep_obj (DyRep): The DyRep object for shared state.
            chunks (list): List of data chunks.

        Returns:
            Results from the processor, retrieved from the results queue.
        """
        # Create threads for analyzer and processor
        analyzer_thread = threading.Thread(
            target=self._run_analyzer,
            args=(analyzer_func, chunks, dyrep_obj)
        )
        processor_thread = threading.Thread(
            target=self._run_processor,
            args=(processor_func, dyrep_obj, chunks)
        )

        # Start threads
        analyzer_thread.start()
        processor_thread.start()

        # Wait for completion
        analyzer_thread.join()
        processor_thread.join()

        # Retrieve results
        return self.process_results_queue.get()

    def process_events_parallel(self, process_event_func, batch_size, shared_lock=None):
        """
        Process events in parallel using ThreadPoolExecutor or ProcessPoolExecutor.

        Args:
            process_event_func (callable): Function to process a single event.
            batch_size (int): Number of events in the batch.
            shared_lock (threading.Lock, optional): Lock for shared resource access.

        Returns:
            List of results, preserving event order.
        """
        results = [None] * batch_size
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        with executor_class(max_workers=self.max_event_workers) as executor:
            # Submit tasks
            future_to_index = {executor.submit(self._wrap_event_func, process_event_func, idx, shared_lock): idx
                               for idx in range(batch_size)}
            # Collect results
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()

        return results

    def _wrap_event_func(self, process_event_func, idx, shared_lock):
        """
        Wrapper for event processing to handle lock acquisition if needed.
        """
        if shared_lock and self.use_processes:
            # Locks can't be passed to processes; assume thread-safe or handle differently
            return process_event_func(idx)
        elif shared_lock:
            with shared_lock:
                return process_event_func(idx)
        else:
            return process_event_func(idx)