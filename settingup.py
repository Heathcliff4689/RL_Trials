# Setting up here!
class ArgumentsChannel:
    def __init__(self):
        # BS antennas
        self.N_bs = 64
        # BS　radio frequency, equal to groups numbers
        self.N_rf = 7
        # frequency of wave
        self.F_wave = 30e9
        # multipaths of signal
        self.multipaths = 7
        # random seeds
        self.seed = None

args = ArgumentsChannel()
