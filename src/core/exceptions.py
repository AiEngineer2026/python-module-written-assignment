"""Domain-specific exceptions for the ideal-function assignment project."""


class AssignmentError(Exception):
    """Base class for assignment-specific errors."""


class DataValidationError(AssignmentError):
    """Raised when input data does not match the expected assignment schema."""


class MissingColumnError(DataValidationError):
    """Raised when a required DataFrame column is missing."""


class UnknownXValueError(DataValidationError):
    """Raised when a test point cannot be matched to an ideal-function x-value."""
