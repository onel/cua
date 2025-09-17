from typing import List, Dict, Any, Tuple, Union
import logging
import signal
from contextlib import contextmanager
from pathlib import Path
import easyocr
from PIL import Image
import numpy as np
import torch

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Exception raised when an OCR operation exceeds the specified timeout duration."""
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager that enforces a timeout for operations using SIGALRM.
    
    Args:
        seconds: Maximum number of seconds to allow the operation to run
        
    Yields:
        None: Context for the timed operation
        
    Raises:
        TimeoutException: If the operation exceeds the specified timeout
        
    Note:
        Only works in the main thread due to signal limitations. In non-main threads,
        the timeout is disabled and a warning is logged.
    """
    import threading
    
    # Check if we're in the main thread
    if threading.current_thread() is threading.main_thread():
        def timeout_handler(signum, frame):
            raise TimeoutException("OCR process timed out")

        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)

        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    else:
        # In a non-main thread, we can't use signal
        logger.warning("Timeout function called from non-main thread; signal-based timeout disabled")
        try:
            yield
        finally:
            pass


class OCRProcessor:
    """Handles optical character recognition (OCR) text detection using EasyOCR.
    
    This class provides text detection capabilities for images, automatically selecting
    the best available device (CUDA, MPS, or CPU) and managing a shared EasyOCR reader
    instance for efficiency.
    """

    _shared_reader = None  # Class-level shared reader instance

    def __init__(self):
        """Initialize the OCR processor with optimal device selection.
        
        Automatically detects and configures the best available computing device
        (CUDA GPU, Apple MPS, or CPU) for OCR processing.
        """
        self.reader = None
        # Determine best available device
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            self.device = "mps"
        logger.info(f"OCR processor initialized with device: {self.device}")

    def _ensure_reader(self):
        """Initialize or retrieve the EasyOCR reader instance.

        Uses a class-level cached reader to avoid reinitializing on every instance,
        improving performance when multiple OCRProcessor instances are used.
        
        Raises:
            RuntimeError: If EasyOCR initialization fails
        """
        # First check if we already have a class-level reader
        if OCRProcessor._shared_reader is not None:
            self.reader = OCRProcessor._shared_reader
            return

        # Otherwise initialize a new one
        if self.reader is None:
            try:
                logger.info("Initializing EasyOCR reader...")
                import easyocr

                # Use GPU if available
                use_gpu = self.device in ["cuda", "mps"]
                self.reader = easyocr.Reader(["en"], gpu=use_gpu)
                
                # Verify reader initialization
                if self.reader is None:
                    raise ValueError("Failed to initialize EasyOCR reader")

                # Cache the reader at class level
                OCRProcessor._shared_reader = self.reader

                logger.info(f"EasyOCR reader initialized successfully with GPU={use_gpu}")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR reader: {str(e)}")
                # Set to a placeholder that will be checked
                self.reader = None
                raise RuntimeError(f"EasyOCR initialization failed: {str(e)}") from e

    def detect_text(
        self, image: Image.Image, confidence_threshold: float = 0.5, timeout_seconds: int = 5
    ) -> List[Dict[str, Any]]:
        """Detect and extract text from an image using EasyOCR.

        Processes the input image to identify text regions, returning normalized
        bounding boxes and confidence scores for detected text elements.

        Args:
            image: PIL Image object to process for text detection
            confidence_threshold: Minimum confidence score (0.0-1.0) for including
                detected text in results
            timeout_seconds: Maximum time in seconds to wait for OCR processing
                before timing out

        Returns:
            List of dictionaries containing detected text information. Each dictionary
            includes:
            - type: Always "text" for text detections
            - bbox: Normalized bounding box coordinates [x1, y1, x2, y2] (0.0-1.0)
            - content: The detected text string
            - confidence: Confidence score for the detection
            - interactivity: Always False for text elements
            
            Returns empty list if OCR fails or times out.
        """
        try:
            # Try to initialize reader, catch any exceptions
            try:
                self._ensure_reader()
            except Exception as e:
                logger.error(f"Failed to initialize OCR reader: {str(e)}")
                return []

            # Ensure reader was properly initialized
            if self.reader is None:
                logger.error("OCR reader is None after initialization")
                return []

            # Convert PIL Image to numpy array
            image_np = np.array(image)

            try:
                with timeout(timeout_seconds):
                    results = self.reader.readtext(
                        image_np, paragraph=False, text_threshold=confidence_threshold
                    )
            except TimeoutException:
                logger.warning("OCR timed out")
                return []
            except Exception as e:
                logger.warning(f"OCR failed: {str(e)}")
                return []

            detections = []
            img_width, img_height = image.size

            for box, text, conf in results:
                # Ensure conf is float
                conf_float = float(conf)
                if conf_float < confidence_threshold:
                    continue

                # Convert box format to [x1, y1, x2, y2]
                # Ensure box points are properly typed as float
                x1 = min(float(point[0]) for point in box) / img_width
                y1 = min(float(point[1]) for point in box) / img_height
                x2 = max(float(point[0]) for point in box) / img_width
                y2 = max(float(point[1]) for point in box) / img_height

                detections.append(
                    {
                        "type": "text",
                        "bbox": [x1, y1, x2, y2],
                        "content": text,
                        "confidence": conf,
                        "interactivity": False,  # Text is typically non-interactive
                    }
                )

            return detections
        except Exception as e:
            logger.error(f"Unexpected error in OCR processing: {str(e)}")
            return []
