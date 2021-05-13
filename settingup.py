# Setting up here!
class ArgumentsChannel:
    def __init__(self):
        # BS antennas
        self.N_bs = 64
        # BS　radio frequency, equal to groups numbers
        self.N_rf = 7
        # desired user number
        self.desired_user_number = 48
        # frequency of wave
        self.F_wave = 30e9
        # tolerance of power difference, Watts
        self.P_tol = 2e-3
        # maximum of power allocation, Watts
        self.P_max = 24e-3
        # sigma noise
        self.sigma = 0.5
        # multipaths of signal
        self.multipaths = 7
        # random seeds
        self.seed = None


args = ArgumentsChannel()
