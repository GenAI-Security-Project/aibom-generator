import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def cleanup_old_files(directory, max_age_days=7):
    """
    Remove files older than max_age_days from the specified directory.
    Optimized to use os.scandir for better performance.
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return 0
    
    removed_count = 0
    now = datetime.now()
    cutoff_time = now - timedelta(days=max_age_days)
    
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    try:
                        # entry.stat().st_mtime is faster than os.path.getmtime
                        file_mtime = datetime.fromtimestamp(entry.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            os.remove(entry.path)
                            removed_count += 1
                            logger.info(f"Removed old file: {entry.path}")
                    except OSError as e:
                        logger.error(f"Error accessing/removing file {entry.path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleanup completed: removed {removed_count} files older than {max_age_days} days from {directory}")
        return removed_count
    except Exception as e:
        logger.error(f"Error during cleanup of directory {directory}: {e}")
        return 0

def limit_file_count(directory, max_files=1000):
    """
    Ensure no more than max_files are kept in the directory (removes oldest first).
    Optimized to use os.scandir.
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return 0
    
    try:
        files = []
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    files.append((entry.path, entry.stat().st_mtime))
        
        # If we are within limits, return early
        if len(files) <= max_files:
            return 0

        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])
        
        # Remove oldest files if we exceed the limit
        files_to_remove = files[:-max_files]
        removed_count = 0
        
        for file_path, _ in files_to_remove:
            try:
                os.remove(file_path)
                removed_count += 1
                logger.info(f"Removed excess file: {file_path}")
            except OSError as e:
                logger.error(f"Error removing file {file_path}: {e}")
        
        logger.info(f"File count limit enforced: removed {removed_count} oldest files, keeping max {max_files}")
        return removed_count
    except Exception as e:
        logger.error(f"Error during file count limiting in directory {directory}: {e}")
        return 0

def perform_cleanup(directory, max_age_days=7, max_files=1000):
    """Perform both time-based and count-based cleanup."""
    time_removed = cleanup_old_files(directory, max_age_days)
    count_removed = limit_file_count(directory, max_files)
    return time_removed + count_removed
