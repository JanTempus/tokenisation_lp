
class tokenInstance:
    token: str
    start: int
    end: int
    lp_value:float
    token_index:int

    def __init__(self,token,start,end,token_index=0,lp_value=float(-1)):
        self.token=token
        self.start=start
        self.end=end
        self.token_index=token_index
        self.lp_value=lp_value
        
    
    def __eq__(self, other):
        if not isinstance(other,possibleToken):
            return False
        return self.token==other.token
    
    def __hash__(self):
        return hash(self.token)

    def __str__(self):
        return f"{self.start,self.end,self.token,self.token_index,self.lp_value}"

    def __repr__(self):
        return self.__str__()

class possibleToken:
    token:str
    lp_value:float
    token_instance_count:int
    token_index:int

    def __init__(self,token):
        self.token=token
        self.lp_value=float(-1)
        self.token_instance_count=0
        self.token_index=0

    def __eq__(self, other):
        if not isinstance(other,possibleToken):
            return False
        return self.token==other.token

    def __hash__(self):
        return hash(self.token)
    
    def __str__(self):
        return f"{self.token, self.lp_value, self.token_instance_count}"

    def __repr__(self):
        return self.__str__()

