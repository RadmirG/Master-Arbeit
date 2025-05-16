from fenics import *
import numpy as np


class PythonFunctionExpression(UserExpression):
    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def eval(self, values, x):
        values[0] = self.func(*x)

    def value_shape(self):
        return ()


class ForwardHeatSolver:
    def __init__(self, domain, a_expr, f_expr, bc, uD=None, uI=None, degree=2):
        self.f = None
        self.a = None
        #self.mesh_domain = None
        self.V = None
        self.mesh = None
        self.dt = None
        self.T = None
        self.is_time_dependent = None
        self.is_two_dim = None

        self.bc = bc
        self.u_D = uD
        self.u_i = uI
        self.domain = domain
        self.degree = degree
        self.a_expr = a_expr
        self.f_expr = f_expr

        self.set_dimensionality()

    def set_dimensionality(self):
        # Detect dimensionality
        self.is_two_dim = True if self.domain['y_domain'] is not None else False
        self.is_time_dependent = True if self.domain['t_domain'] is not None else False

        if not self.is_two_dim:
            self.build_1D()
        else:
            self.build_2D()

        if self.is_time_dependent:
            t_start, t_end, nT = self.domain['t_domain']
            self.T = t_end - t_start
            self.dt = (t_end - t_start)/nT

    def wrap_expr(self, input_expr):
        if isinstance(input_expr, (Expression, UserExpression)):
            return input_expr
        elif callable(input_expr):
            return PythonFunctionExpression(func=input_expr, degree=self.degree)
        else:
            raise TypeError("Input must be a FEniCS Expression or a Python function.")

    def build_1D(self):
        x_start, x_end, nx = self.domain['x_domain']
        self.mesh = IntervalMesh(nx, x_start, x_end)
        self.V = FunctionSpace(self.mesh, "P", self.degree)
        #self.mesh_domain = self.V.tabulate_dof_coordinates().flatten()
        self.set_conditions_and_coefficients_1D()

    def build_2D(self):
        x_start, x_end, nx = self.domain['x_domain']
        y_start, y_end, ny = self.domain['y_domain']
        self.mesh = RectangleMesh(Point(x_start, y_start), Point(x_end, y_end), nx, ny)
        self.V = FunctionSpace(self.mesh, "P", self.degree)
        #self.mesh_domain = self.V.tabulate_dof_coordinates()
        self.set_conditions_and_coefficients_2D()

    def set_conditions_and_coefficients_1D(self):
        self.boundary_condition = self.wrap_expr(self.bc)
        self.bc = DirichletBC(self.V, self.boundary_condition, "on_boundary")
        if self.u_D is not None:
            self.u_D = Constant(0.0)
        if self.u_i is not None:
            self.initial_condition = self.wrap_expr(self.u_i)
            self.u_i = interpolate(self.initial_condition, self.V)

        self.a = self.wrap_expr(self.a_expr)
        self.f = self.wrap_expr(self.f_expr)

    def set_conditions_and_coefficients_2D(self):
        self.u_D = Constant(0.0)
        self.bc = DirichletBC(self.V, self.u_D, "on_boundary")
        self.u_i = interpolate(Constant(0.0), self.V)

        self.a = self.wrap_expr(self.a_expr)
        self.f = self.wrap_expr(self.f_expr)

    def solve(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        u_ = Function(self.V)


        snapshots = []
        if not self.is_time_dependent:
            F = self.a * dot(grad(u), grad(v)) * dx - self.f * v * dx
            a_form, L_form = lhs(F), rhs(F)
            solve(a_form == L_form, u_, self.bc)
            snapshots.append(u_.copy(deepcopy=True))
        else:
            F = ((u - self.u_i) / self.dt) * v * dx + self.a * dot(grad(u), grad(v)) * dx - self.f * v * dx
            a_form, L_form = lhs(F), rhs(F)
            time = 0.0

            while time < self.T:
                time += self.dt
                try:
                    self.f.t = time  # if f is time-dependent Expression
                except AttributeError:
                    pass  # OK for time-independent functions
                solve(a_form == L_form, u_, self.bc)
                self.u_i.assign(u_)
                snapshots.append(u_.copy(deepcopy=True))

        coords = self.mesh.coordinates().flatten()  # Mesh vertex
        sol = u_.compute_vertex_values(self.mesh)
        return sol, coords
