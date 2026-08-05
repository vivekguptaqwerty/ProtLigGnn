from typing import Optional

class ProteinParserError(Exception):
    def __init__(self, file_path: str, parser_name: str, reason: str, recovery_suggestion: str):
        self.file_path = file_path
        self.parser_name = parser_name
        self.reason = reason
        self.recovery_suggestion = recovery_suggestion
        super().__init__(
            f"Parser '{parser_name}' failed on file '{file_path}'. Reason: {reason}. "
            f"Recovery suggestion: {recovery_suggestion}"
        )

class InvalidStructureError(ProteinParserError):
    pass

class UnsupportedFormatError(ProteinParserError):
    pass

class CorruptedFileError(ProteinParserError):
    pass

class EmptyStructureError(ProteinParserError):
    pass

class CoordinateError(ProteinParserError):
    pass

class ParserRegistrationError(Exception):
    def __init__(self, parser_name: str, reason: str):
        self.parser_name = parser_name
        self.reason = reason
        super().__init__(f"Failed to register parser '{parser_name}'. Reason: {reason}")
