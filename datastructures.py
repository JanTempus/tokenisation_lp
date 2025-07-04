
class tokenInstance:
    token: str
    start: int
    end: int
    lpValue:float

    def __init__(self,token,start,end):
        self.token=token
        self.start=start
        self.end=end
        self.lpValue=float(-1)
    
    def __eq__(self, other):
        if not isinstance(other,possibleToken):
            return False
        return self.token==other.token
    
    def __hash__(self):
        return hash(self.token)

    def __str__(self):
        return f"{self.start,self.end,self.token,self.lpValue}"

    def __repr__(self):
        return self.__str__()
    def to_dict(self):
        return {
            "token": self.token,
            "start": self.start,
            "end": self.end,
            "lp_value": self.lpValue
        }

class possibleToken:
    token:str
    lpValue:float

    def __init__(self,token):
        self.token=token
        self.lpValue=float(-1)

    def __eq__(self, other):
        if not isinstance(other,possibleToken):
            return False
        return self.token==other.token

    def __hash__(self):
        return hash(self.token)
    
    def __str__(self):
        return f"{self.token, self.lpValue}"

    def __repr__(self):
        return self.__str__()
    def to_dict(self):
        return {
            "token": self.token,
            "lp_value": self.lpValue
        }

