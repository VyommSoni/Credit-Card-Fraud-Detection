import os
import sys

class CreditCradFraudDetection(Exception):
    def __init__(self, error_message, error_details: sys):
        self.error_message = self.get_detailed_error_message(
            error_message,
            error_details
        )

        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(error_message, error_details: sys):
        _, _, exc_tb = error_details.exc_info()

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return (
            f"Error occurred in script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"error message: [{error_message}]"
        )

    def __str__(self):
        return self.error_message