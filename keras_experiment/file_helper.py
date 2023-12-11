import json


class FileHelper:
    @staticmethod
    def read_json_file(filename: str):
        with open(filename) as f:
            data = json.load(f)
        return data

    @staticmethod
    def read_csv_file(filename: str):
        with open(filename) as f:
            data = f.readlines()
        data = [x.split(',') for x in data]
        return data