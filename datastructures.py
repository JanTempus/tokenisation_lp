
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

class possibleToken:
    token:str
    lpValue:float
    token_instance_count:int

    def __init__(self,token):
        self.token=token
        self.lpValue=float(-1)
        self.token_instance_count=0

    def __eq__(self, other):
        if not isinstance(other,possibleToken):
            return False
        return self.token==other.token

    def __hash__(self):
        return hash(self.token)
    
    def __str__(self):
        return f"{self.token, self.lpValue, self.token_instance_count}"

    def __repr__(self):
        return self.__str__()

