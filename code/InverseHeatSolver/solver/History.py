# ======================================================================================================================
# This class handles the Loss corves of InverseHeatSolver
# ======================================================================================================================
# Radmir Gesler, 2024, master thesis at BHT Berlin by Prof. Dr. Frank Haußer
# ======================================================================================================================

class History:
    def __init__(self):
        self.losses = {'loss': [],
                       'pde_loss': [],
                       'a_grad_loss': [],
                       'gPINN_loss': []}
        self.steps = []

    def append_loss(self, loss):
        self.losses['loss'].append(loss)

    def append_pde_loss(self, pde_loss):
        self.losses['pde_loss'].append(pde_loss)

    def append_a_grad_loss(self, a_grad_loss):
        self.losses['a_grad_loss'].append(a_grad_loss)

    def append_gPINN_loss(self, gPINN_loss):
        self.losses['gPINN_loss'].append(gPINN_loss)