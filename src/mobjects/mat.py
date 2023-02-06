from manim import *
import numpy as np
import itertools as it

# TODO: inherit from Matrix instead


class Mat(VMobject):
    def __init__(self, mat: np.ndarray, hscale: float = 1.75, rounding: int = 1,  **kwargs):
        super().__init__(**kwargs)
        self._mat = mat

        self.rounding = rounding
        self.hscale = hscale

        self._update_matrix()
        self.add(self.matrix)

    def _update_matrix(self):
        if (self.mat.astype(np.int64) == self.mat).all():
            self.h_buff = max(MathTex(int(x)).length_over_dim(0)
                              for x in self.mat.flatten()) * self.hscale

            self.matrix = IntegerMatrix(self.mat, h_buff=self.h_buff)
        else:
            self.h_buff = max(MathTex(x).length_over_dim(0)
                              for x in self.mat.flatten()) * self.hscale
            self.matrix = Matrix(
                np.round(self.mat, self.rounding), h_buff=self.h_buff)

    def rect_elem(self, i: int, j: int) -> SurroundingRectangle:
        return SurroundingRectangle(self.matrix.get_rows()[i][j])

    def rect_row(self, i: int) -> SurroundingRectangle:
        return SurroundingRectangle(self.matrix.get_rows()[i])

    def rect_col(self, j: int) -> SurroundingRectangle:
        return SurroundingRectangle(self.matrix.get_columns()[j])

    def rect_around(self, i_from: int, j_from: int, i_to: int, j_to: int) -> SurroundingRectangle:
        rows = self.matrix.get_rows()
        return SurroundingRectangle(VGroup(*(rows[i][j_from:j_to] for i in range(i_from, i_to))))

    def __getitem__(self, *args):
        return self.mat.__getitem__(*args)

    def __setitem__(self, key, value):
        self.remove(self.matrix)
        self.mat.__setitem__(key, value)

        self._update_matrix()
        # self.h_buff = max(self.h_buff,
        #                   MathTex(value).length_over_dim(0) * self.hscale)

        # if (self.mat.astype(np.int64) == self.mat).all():
        #     self.matrix = IntegerMatrix(self.mat, h_buff=self.h_buff)
        # else:
        #     self.matrix = Matrix(
        #         np.round(self.mat, self.rounding), h_buff=self.h_buff)

        self.add(self.matrix)

    @property
    def mat(self) -> np.ndarray:
        return self._mat

    @mat.setter
    def mat(self, new: np.ndarray):
        self._mat = new
