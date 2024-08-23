class IState:
    def __init__(self) -> None:
        self.link={}
    def Awake(self):
        pass
    def Update(self):
        pass
    def Destroy(self):
        pass
    def Link(self,state,func):
        self.link[state] = func


class FSM:
    
    def __init__(self,state:IState):
        self.current=state
        pass
    def update(self):
        self.CheckLink(self.current)
        self.current.Update()
        
        pass
    def CheckLink(self,current:IState):
        for i,func in current.link.items():
            if func(current):
                self.current.Destroy()
                self.current:IState = i
                self.current.Awake()
                break
    
    

        