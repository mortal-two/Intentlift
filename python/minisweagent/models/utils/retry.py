"""Retry utility for model queries."""

import logging
from collections.abc import Callable

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def retry(
    *,
    logger: logging.Logger,
    abort_exceptions: list[type[Exception]],
    max_attempts: int = 1,
    retry_predicate: Callable[[BaseException], bool] | None = None,
) -> Retrying:
    """Thin wrapper around tenacity.Retrying to make use of global config etc.

    Args:
        logger: Logger to use for reporting retries
        abort_exceptions: Exceptions to abort on.
        retry_predicate: Optional predicate limiting retries to transient
            exceptions.  When omitted, the historical policy retries every
            exception that is not in ``abort_exceptions``.

    Returns:
        A tenacity.Retrying object.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if retry_predicate is None:
        retry_condition = retry_if_not_exception_type(tuple(abort_exceptions))
    else:
        abort_types = tuple(abort_exceptions)
        retry_condition = retry_if_exception(
            lambda exc: (
                not isinstance(exc, abort_types) and retry_predicate(exc)
            )
        )
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_condition,
    )
