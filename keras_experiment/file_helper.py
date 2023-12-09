import json


class FileHelper:
    @staticmethod
    def read_json_file(filename: str):
        with open(filename) as f:
            data = json.load(f)
        return data