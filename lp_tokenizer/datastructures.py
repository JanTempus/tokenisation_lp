
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

    def __init__(self,token:str,lp_value:int=float(-1),instance_count:int=0,index:int=0):
        self.token=token
        self.lp_value=lp_value
        self.token_instance_count=instance_count
        self.token_index=index

    def get_token(self):
        return self.token
    
    def get_count(self):
        return self.token_instance_count
    
    def get_index(self):
        return self.token_index

    def __eq__(self, other):
        if not isinstance(other,possibleToken):
            return False
        return self.token==other.token

    def __hash__(self):
        return hash(self.token)
    
    def __str__(self):
        return f"{self.token, self.lp_value, self.token_instance_count,self.token_index}"

    def __repr__(self):
        return self.__str__()

