import pandas as pd


class MergeEngine:
    """
    Responsible for merging all project datasets into one master table.
    """

    def __init__(self, datasets):

        self.student_info = datasets["student_info"]

        self.student_registration = datasets["student_registration"]

        self.student_assessments = datasets["student_assessments"]

        self.assessments = datasets["assessments"]

        self.student_vle = datasets["student_vle"]

        self.vle = datasets["vle"]

        self.courses = datasets["courses"]

    def merge_registration(self):

        merged = self.student_info.merge(

            self.student_registration,

            on=[
                "code_module",
                "code_presentation",
                "id_student"
            ],

            how="left"
        )

        assert merged.shape[0] == self.student_info.shape[0], \
            "Registration merge created duplicate rows."

        return merged