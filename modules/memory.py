import time


class ObjectMemory:

    def __init__(self):

        self.memory = {}

        self.cooldown = 5

    def should_speak(self, obj):

        current = time.time()

        if obj not in self.memory:

            self.memory[obj] = current
            return True

        if current - self.memory[obj] > self.cooldown:

            self.memory[obj] = current
            return True

        return False