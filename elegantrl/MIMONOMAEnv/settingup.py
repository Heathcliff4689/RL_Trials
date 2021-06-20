# Setting up here!
class ArgumentsChannel:
    def __init__(self):
        # BS antennas
        self.N_bs = 128
        # BS　radio frequency, equal to groups numbers
        self.N_rf = 12
        # desired user number
        self.desired_user_number = 32
        # frequency of wave
        self.F_wave = 30e9
        # tolerance of power difference, Watts
        self.P_tol = 2e-3
        # maximum of power allocation, Watts
        self.P_min = self.P_tol
        self.P_max = 24e-3
        # sigma noise, should be considered extensively?
        self.sigma = 0.1
        # analog bf discount, self.abf_discount = {12: 43, 16: 55, 32: 100, 64: 150, 128: 300}
        self.abf_discount = {12: 4.3, 16: 5.5, 32: 10, 64: 15, 128: 30}
        # multipaths of signal
        self.multipaths = 7
        # random seeds, should be considered extensively?
        self.seed = None


args_channel = ArgumentsChannel()
