This is a fullframework for Reinforcement Learning that allows a developer to develop algorithms in straightforward manner.
Ex.


class TD(MRP):
    # def stop_exp(self):
        
    # ----------------------------- 🌖 online learning ----------------------    
    def online(self, s, rn,sn, done, *args): 
        self.V[s] += self.α*(rn + (1- done)*self.γ*self.V[sn] - self.V[s])


