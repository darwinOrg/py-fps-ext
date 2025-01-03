class Result:
    def __init__(self, success=True, code=None, message=None, data=None):
        self.success = success
        self.code = code
        self.message = message
        self.data = data

    @staticmethod
    def success(data=None):
        return Result(success=True, code=200, data=data)

    @staticmethod
    def fail(message, code=1):
        return Result(success=False, code=code, message=message)

    def to_dict(self):
        return {
            "success": self.success,
            "code": self.code,
            "message": self.message,
            "data": self.data
        }