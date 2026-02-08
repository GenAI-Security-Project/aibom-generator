import time
from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import asyncio  # Concurrency limiting

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app, 
        rate_limit_per_minute=10,
        rate_limit_window=60,
        protected_routes=["/generate", "/api/generate", "/api/generate-with-report"]
    ):
        super().__init__(app)
        self.rate_limit_per_minute = rate_limit_per_minute
        self.rate_limit_window = rate_limit_window
        self.protected_routes = protected_routes
        self.ip_requests = defaultdict(list)
        logger.info(f"Rate limit middleware initialized: {rate_limit_per_minute} requests per {rate_limit_window}s")
        
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Only apply rate limiting to protected routes
        if any(request.url.path.startswith(route) for route in self.protected_routes):
            # Clean up old requests for this IP
            self.ip_requests[client_ip] = [t for t in self.ip_requests[client_ip] 
                                          if current_time - t < self.rate_limit_window]
            
            # Periodic cleanup of all IPs (every ~100 requests to avoid overhead)
            # In a production app, use a background task or Redis
            if len(self.ip_requests) > 1000 and hash(client_ip) % 100 == 0:
                 self._cleanup_all_ips(current_time)

            # Check if rate limit exceeded
            if len(self.ip_requests[client_ip]) >= self.rate_limit_per_minute:
                logger.warning(f"Rate limit exceeded for IP {client_ip} on {request.url.path}")
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please try again later."}
                )
            
            # Add current request timestamp
            self.ip_requests[client_ip].append(current_time)
        
        # Process the request
        response = await call_next(request)
        return response

    def _cleanup_all_ips(self, current_time):
        """Remove IPs that haven't made requests in the window"""
        to_remove = []
        for ip, timestamps in self.ip_requests.items():
            # If latest timestamp is older than window, remove IP
            if not timestamps or (current_time - timestamps[-1] > self.rate_limit_window):
                to_remove.append(ip)
        for ip in to_remove:
            del self.ip_requests[ip]

class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app, 
        max_concurrent_requests=5,
        timeout=5.0,
        protected_routes=None
    ):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.timeout = timeout
        self.protected_routes = protected_routes or ["/generate", "/api/generate", "/api/generate-with-report"]
        logger.info(f"Concurrency limit middleware initialized: {max_concurrent_requests} concurrent requests")
        
    async def dispatch(self, request, call_next):
        try:
            # Only apply to protected routes
            if any(request.url.path.startswith(route) for route in self.protected_routes):
                try:
                    # Try to acquire the semaphore
                    acquired = False
                    try:
                        # Use wait_for instead of timeout context manager for compatibility
                        await asyncio.wait_for(self.semaphore.acquire(), timeout=self.timeout)
                        acquired = True
                        return await call_next(request)
                    finally:
                        if acquired:
                            self.semaphore.release()
                except asyncio.TimeoutError:
                    # Timeout waiting for semaphore
                    logger.warning(f"Concurrency limit reached for {request.url.path}")
                    return JSONResponse(
                        status_code=503, 
                        content={"detail": "Server is at capacity. Please try again later."}
                    )
            else:
                # For non-protected routes, proceed normally
                return await call_next(request)
        except Exception as e:
            logger.error(f"Error in ConcurrencyLimitMiddleware: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Internal server error in middleware: {str(e)}"}
            )


# Protection against large request payloads
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_content_length=1024*1024):  # 1MB default
        super().__init__(app)
        self.max_content_length = max_content_length
        logger.info(f"Request size limit middleware initialized: {max_content_length} bytes")
        
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get('content-length')
        if content_length:
            if int(content_length) > self.max_content_length:
                logger.warning(f"Request too large: {content_length} bytes")
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request too large"}
                )
        return await call_next(request)
