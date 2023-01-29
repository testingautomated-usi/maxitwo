from itertools import count


class IdGenerator:
    __instance: "IdGenerator" = None

    @staticmethod
    def get_instance(start_count: int = 1):
        if IdGenerator.__instance is None:
            IdGenerator(start_count=start_count)
        return IdGenerator.__instance

    def __init__(self, start_count: int = 1):

        if IdGenerator.__instance is not None:
            raise Exception("This class is a singleton!")

        self.id_counter = count(start=start_count)
        IdGenerator.__instance = self

    def get_id(self) -> int:
        return next(self.id_counter)
