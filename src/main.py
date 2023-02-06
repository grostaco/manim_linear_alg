from manim import *
import numpy as np
from mobjects.mat import Mat


def pivot_selection(scene: Scene, mat: np.ndarray):
    matrix = Matrix(mat.round(1))
    text = Text('Pivot selection')
    annot = Text('Pivot is zero; searching for a non-zero pivot',
                 font_size=DEFAULT_FONT_SIZE * 0.5)

    (m, n) = mat.shape

    text.next_to(matrix, DOWN * 2)

    scene.play(Write(VGroup(matrix, text).to_corner(UL)))

    col_rect = SurroundingRectangle(matrix.get_columns()[0])
    scene.play(Create(col_rect))

    target = col_rect.generate_target()

    for col in range(min(m, n)):
        if col != 0:
            target.move_to(matrix.get_columns()[col])
        scene.play(MoveToTarget(col_rect))

        if mat[col][col] == 0:
            elem_rect = SurroundingRectangle(
                matrix.get_entries()[col * n + col])
            scene.remove(col_rect)
            scene.play(TransformFromCopy(col_rect, elem_rect))

            annot.next_to(matrix, RIGHT)
            scene.play(FadeIn(annot))
            scene.remove(annot)
            scene.play(FadeOut(annot))
            scene.remove(elem_rect)

            scol = 0 if col != 0 else 1
            row_rect = SurroundingRectangle(
                matrix.get_rows()[scol])
            row_target = row_rect.generate_target()
            scene.play(TransformFromCopy(elem_rect, row_rect))

            for row in range(scol, n):
                if col != row:
                    row_target.move_to(matrix.get_rows()[row])

                    if row != scol:
                        scene.play(MoveToTarget(row_rect))

                    if mat[row][col] != 0:
                        swap = Text(
                            f'Swap row {row} with row {col}', font_size=DEFAULT_FONT_SIZE * 0.5)
                        swap.next_to(text, DOWN)
                        scene.play(FadeIn(swap))
                        scene.remove(swap)
                        scene.play(FadeOut(swap))

                        if mat[col][row] == 0 and row < col:
                            continue

                        mat[[col, row]] = mat[[row, col]]
                        new_matrix = Matrix(mat.round(1)).move_to(matrix)
                        scene.remove(matrix)
                        scene.play(TransformFromCopy(
                            matrix, new_matrix))

                        matrix = new_matrix
                        break

            scene.remove(row_rect)

            if col + 1 < n:
                col_rect.move_to(matrix.get_columns()[col+1])
                scene.play(TransformFromCopy(row_rect, col_rect))

    return mat


def row_reduce(scene: Scene, mat: np.ndarray):
    matrix = Matrix(mat.round(1), h_buff=1.6)
    text = Text('Row Reduction')

    text.next_to(matrix, DOWN * 2)
    scene.play(Write(VGroup(matrix, text).to_edge(UL)))

    m, n = mat.shape

    pivot_rect = SurroundingRectangle(matrix.get_entries()[0]).set_color(RED)
    target_rect = SurroundingRectangle(
        matrix.get_entries()[0]).set_color(BLUE)

    for col in range(min(m, n)):
        pivot = matrix.get_entries()[col * n + col]
        pivot_rect = SurroundingRectangle(pivot).set_color(RED)
        target_rect = SurroundingRectangle(pivot).set_color(RED)

        scene.play(Create(pivot_rect))

        for row in range(col + 1, min(m, n)):
            target = matrix.get_entries()[row * n + col]
            target_rect.target = SurroundingRectangle(target).set_color(BLUE)
            scene.play(MoveToTarget(target_rect))

            if mat[row][col] != 0:
                scale = mat[row][col] / mat[col][col]

                scale_tex = MathTex(str(round(scale, 3)))

                eq = MathTex(f'l_{{{row+1}{col+1}}} =')
                (rhs1, over, rhs2) = MathTex(
                    f'{round(mat[row][col], 3)}', '\\over', f'{round(mat[col][col], 3)}')

                rhs1.set_color(BLUE)
                rhs2.set_color(RED)

                rhs = VGroup(rhs1, over, rhs2)
                rhs.next_to(eq, RIGHT)
                VGroup(eq, rhs1, over, rhs2).next_to(
                    matrix, aligned_edge=UR, direction=RIGHT * 2)

                scene.play(TransformFromCopy(VGroup(target, pivot), rhs),
                           Write(eq))

                scale_tex.next_to(eq, RIGHT)
                scene.play(Transform(rhs, scale_tex,
                           replace_mobject_with_target_in_scene=True))

                mat[row] -= mat[col] * scale
                mat[row][0:col+1] = abs(mat[row][0:col+1])

                new_matrix = Matrix(mat.round(1), h_buff=1.6).to_corner(UL)
                text.target = Text("Row Reduction").next_to(
                    new_matrix, DOWN * 2)

                pivot_rect.target = SurroundingRectangle(
                    new_matrix.get_entries()[col * n + col]).set_color(RED)
                target_rect.target = SurroundingRectangle(
                    new_matrix.get_entries()[row * n + col]).set_color(BLUE)

                eq.generate_target()
                scale_tex.generate_target()

                eq.target.next_to(
                    new_matrix, aligned_edge=UR, direction=RIGHT * 2)
                scale_tex.target.next_to(eq.target, RIGHT)

                scene.play(Transform(matrix, new_matrix, replace_mobject_with_target_in_scene=True),
                           MoveToTarget(text),
                           MoveToTarget(pivot_rect), MoveToTarget(target_rect),
                           MoveToTarget(eq), MoveToTarget(scale_tex))

                matrix = new_matrix

                scene.play(Unwrite(VGroup(eq, scale_tex)), run_time=0.5)
                scene.remove(eq, scale_tex)

        scene.play(Uncreate(pivot_rect), Uncreate(target_rect))


def pivot_row_reduce(scene: Scene, mat: np.ndarray):
    matrix = Matrix(mat.round(1))
    pivot_text = Text('Pivot selection')
    annot = Text('Pivot is zero; searching for a non-zero pivot',
                 font_size=DEFAULT_FONT_SIZE * 0.5)

    (m, n) = mat.shape

    pivot_text.next_to(matrix, DOWN * 2)

    pivot_vg = VGroup(matrix, pivot_text).to_corner(UL)
    scene.play(Write(pivot_vg))

    col_rect = SurroundingRectangle(matrix.get_columns()[0])
    scene.play(Create(col_rect))

    target = col_rect.generate_target()

    for col in range(min(m, n)):
        if col != 0:
            target.move_to(matrix.get_columns()[col])
        scene.play(MoveToTarget(col_rect))

        if mat[col][col] == 0:
            elem_rect = SurroundingRectangle(
                matrix.get_entries()[col * n + col])
            scene.remove(col_rect)
            scene.play(TransformFromCopy(col_rect, elem_rect))

            annot.next_to(matrix, RIGHT)
            scene.play(FadeIn(annot))
            scene.remove(annot)
            scene.play(FadeOut(annot))
            scene.remove(elem_rect)

            scol = 0 if col != 0 else 1
            row_rect = SurroundingRectangle(
                matrix.get_rows()[scol])
            row_target = row_rect.generate_target()
            scene.play(TransformFromCopy(elem_rect, row_rect))

            for row in range(scol, n-1):
                if col != row:
                    row_target.move_to(matrix.get_rows()[row])

                    if row != scol:
                        scene.play(MoveToTarget(row_rect))

                    if mat[row][col] != 0:
                        swap = Text(
                            f'Swap row {row} with row {col}', font_size=DEFAULT_FONT_SIZE * 0.5)
                        swap.next_to(pivot_text, DOWN)
                        scene.play(FadeIn(swap))
                        scene.remove(swap)
                        scene.play(FadeOut(swap))

                        if mat[col][row] == 0 and row < col:
                            continue

                        mat[[col, row]] = mat[[row, col]]
                        new_matrix = Matrix(mat.round(1)).move_to(matrix)
                        scene.remove(matrix)
                        scene.play(TransformFromCopy(
                            matrix, new_matrix))

                        matrix = new_matrix
                        break

            scene.remove(row_rect)

            if col + 1 < n:
                col_rect.move_to(matrix.get_columns()[col+1])
                scene.play(TransformFromCopy(row_rect, col_rect))

    scene.remove(col_rect)

    text = Text('Row Reduction')
    text.generate_target()

    text.next_to(matrix, DOWN * 2)
    # scene.remove(pivot_vg)
    scene.play(Transform(pivot_text, text,
               replace_mobject_with_target_in_scene=True))

    m, n = mat.shape

    pivot_rect = SurroundingRectangle(matrix.get_entries()[0]).set_color(RED)
    target_rect = SurroundingRectangle(
        matrix.get_entries()[0]).set_color(BLUE)

    for col in range(min(m, n)):
        pivot = matrix.get_entries()[col * n + col]
        pivot_rect = SurroundingRectangle(pivot).set_color(RED)
        target_rect = SurroundingRectangle(pivot).set_color(RED)

        scene.play(Create(pivot_rect))

        for row in range(col + 1, min(m, n)):
            target = matrix.get_entries()[row * n + col]
            target_rect.target = SurroundingRectangle(target).set_color(BLUE)
            scene.play(MoveToTarget(target_rect))

            if mat[row][col] != 0:
                scale = mat[row][col] / mat[col][col]

                scale_tex = MathTex(str(round(scale, 3)))

                eq = MathTex(f'l_{{{row+1}{col+1}}} =')
                (rhs1, over, rhs2) = MathTex(
                    f'{round(mat[row][col], 3)}', '\\over', f'{round(mat[col][col], 3)}')

                rhs1.set_color(BLUE)
                rhs2.set_color(RED)

                rhs = VGroup(rhs1, over, rhs2)
                rhs.next_to(eq, RIGHT)
                VGroup(eq, rhs1, over, rhs2).next_to(
                    matrix, aligned_edge=UR, direction=RIGHT * 2)

                scene.play(TransformFromCopy(VGroup(target, pivot), rhs),
                           Write(eq))

                scale_tex.next_to(eq, RIGHT)
                scene.play(Transform(rhs, scale_tex,
                           replace_mobject_with_target_in_scene=True))

                mat[row] -= mat[col] * scale
                mat[row][0:col+1] = abs(mat[row][0:col+1])

                new_matrix = Matrix(mat.round(1), h_buff=1.6).to_corner(UL)
                text.target = Text("Row Reduction").next_to(
                    new_matrix, DOWN * 2)

                pivot_rect.target = SurroundingRectangle(
                    new_matrix.get_entries()[col * n + col]).set_color(RED)
                target_rect.target = SurroundingRectangle(
                    new_matrix.get_entries()[row * n + col]).set_color(BLUE)

                eq.generate_target()
                scale_tex.generate_target()

                eq.target.next_to(
                    new_matrix, aligned_edge=UR, direction=RIGHT * 2)
                scale_tex.target.next_to(eq.target, RIGHT)

                scene.play(Transform(matrix, new_matrix, replace_mobject_with_target_in_scene=True),
                           MoveToTarget(text),
                           MoveToTarget(pivot_rect), MoveToTarget(target_rect),
                           MoveToTarget(eq), MoveToTarget(scale_tex))

                matrix = new_matrix

                scene.play(Unwrite(VGroup(eq, scale_tex)), run_time=0.5)
                scene.remove(eq, scale_tex)

        scene.play(Uncreate(pivot_rect), Uncreate(target_rect))

    rref_text = Text('RREF')

    scene.play(Transform(text, rref_text.next_to(matrix, DOWN * 2)))

    m, n = mat.shape

    col_rect = SurroundingRectangle(matrix.get_columns()[min(m, n) - 1])
    col_rect.generate_target()

    for col in range(min(m, n) - 1, -1, -1):
        pivot_rect = SurroundingRectangle(
            matrix.get_rows()[col][col]).set_color(RED)
        target_rect = SurroundingRectangle(
            matrix.get_rows()[col][col]).set_color(BLUE)
        scene.play(Create(pivot_rect))

        if mat[col][col] != 0.:
            if mat[col][col] != 1:
                mat[col] /= mat[col][col]

                cmat = mat.copy().round(1)
                cmat[cmat == -.0] = np.abs(cmat[cmat == -.0])
                new_matrix = Matrix(cmat.round(1))

                new_matrix = Matrix(cmat.round(1))
                scene.play(Transform(matrix, new_matrix.to_corner(UL)), Transform(pivot_rect, SurroundingRectangle(
                    new_matrix.get_rows()[col][col]).set_color(RED)))
                target_rect = SurroundingRectangle(
                    matrix.get_rows()[col][col]).set_color(BLUE)

            for row in range(col-1, -1, -1):
                target = matrix.get_rows()[row][col]

                target_rect.target = SurroundingRectangle(
                    target).set_color(BLUE)
                scene.play(MoveToTarget(target_rect))

                if mat[row][col] != 0.:
                    scale = mat[row][col] / mat[col][col]
                    mat[row] -= scale * mat[col]

                    cmat = mat.copy().round(1)
                    cmat[cmat == -.0] = np.abs(cmat[cmat == -.0])
                    new_matrix = Matrix(cmat.round(1))

                    scene.play(Transform(matrix, new_matrix.to_corner(UL)), Transform(target_rect, SurroundingRectangle(
                        new_matrix.get_rows()[row][col]).set_color(BLUE)), Transform(pivot_rect, SurroundingRectangle(
                            new_matrix.get_rows()[col][col]).set_color(RED)))

            scene.play(Uncreate(target_rect), Uncreate(pivot_rect))


def lup_decomposition(scene: Scene, mat: np.ndarray):
    m, n = mat.shape

    l = Mat(mat)
    p = Mat(np.eye(*mat.shape), hscale=2.5)

    pivot_group = VGroup(MathTex("L ="), l, MathTex("P ="), p).arrange()

    text = Text("Partial Pivoting")
    text.next_to(pivot_group, DOWN * 2)

    scene.play(Write(pivot_group), Write(text))

    r = l.rect_elem(0, 0).set_color(RED)
    scene.play(Create(r))
    for j in range(n):
        if j != 0:
            r.target = l.rect_elem(j, j).set_color(RED)
            scene.play(MoveToTarget(r))

        if l[j][j] == 0:
            # Search below
            if j != n - 1:
                c = l.rect_row(j+1).set_color(BLUE)
                scene.play(Transform(r, c))

                for i in range(j + 1, min(n, m)):
                    if i != j + 1:
                        r.generate_target().move_to(l.matrix.get_rows()[i])
                        scene.play(MoveToTarget(r))

                    if l[i][j] != 0:
                        pc = pivot_group.copy()

                        l = pc[1]
                        p = pc[3]

                        l[[i, j]] = l[[j, i]]
                        p[[i, j]] = p[[j, i]]
                        pc.arrange()

                        scene.play(
                            Transform(pivot_group, pc, replace_mobject_with_target_in_scene=True))
                        pivot_group = pc

                        break

    scene.play(Uncreate(r))


class MatDisplay(Scene):
    def construct(self):
        # mat = (np.random.random((4, 5)) * 100).astype(np.float64)
        # mat[np.random.random((4, 5)) < 0.4] = 0.
        mat = np.array([[0.,  12.1225,  54.2164, 33.53767495, 31.24970158],
                        [21.82066998,  0., 53.09296433,
                         41.37424145,  2.72551121],
                        [12.07815709,  32.1253, 42.81924439,  7.69020931,  0.],
                        [21.18154007, 50.1444337,  0., 88.50829655, 77.81459294]])
        mat = np.array([
            [0, -3, 1, 2],
            [3, 0, -5, 6],
            [1, 1, 2, 4],
            [5, 1, 3, -2]
        ], dtype=np.float64)
        #mother = mat.copy()
        #mat[2, 1] = 2142

        # self.play(Write(mat))
        lup_decomposition(self, mat)
        self.wait()

        #pivot_row_reduce(self, mat)

        #mat = row_reduce(self, mat)
        # row_reduce(self, mat)
        # pivot_selection(self, ))

        # mat = IntegerMatrix([[1, 2], [4, 5]])

        # mat2 = MathTex(
        #     """
        #     \\left[
        #         \\begin{array}{ccc}
        #             1 - 4 & 2 \\\\
        #             4 & 5
        #         \\end{array}
        #     \\right]
        #     """)[0]

        # mat3 = IntegerMatrix([[1, 2], [4, 5]])

        # rhs = VGroup(*(mat2[i] for i in (2, 3)))

        # self.play(TransformFromCopy(mat.get_brackets(),
        #                             VGroup(*(mat2[i] for i in [0, -1]))),
        #           TransformFromCopy(mat.get_entries(), VGroup(
        #               *(mat2[i] for i in [1, 4, 5, 6]))),
        #           Write(rhs))

        # rhs.target = mat2[1]

        # self.play(Transform(VGroup(*(mat2[i] for i in [0, -1])),
        #                     mat3.get_brackets()),
        #           Transform(
        #               VGroup(*(mat2[i] for i in [1, 4, 5, 6])), mat3.get_entries()),
        #           FadeOut(rhs, target_position=rhs.target)),

        # TransformFromCopy(VGroup(
        #     *(mat2[i] for i in [1, 4, 5, 6])), mat3.get_entries())
        # self.play(TransformFromCopy(mat, mat2]))

        # self.play(Write(mat))

        # rect = SurroundingRectangle(mat.get_columns()[0])
        # target = rect.generate_target()
        # target.move_to(mat.get_columns()[1])

        # self.play(MoveToTarget(rect))

        # target.move_to(mat.get_columns()[2])
        # self.play(MoveToTarget(rect))

        # if col + 1 < n:
        #     row_rect = SurroundingRectangle(matrix.get_rows()[col+1])
        #     row_target = row_rect.generate_target()
        #     scene.play(TransformFromCopy(elem_rect, row_rect))

        #     for row in range(col+1, n):
        #         if mat[row][col] != 0:
        #             mat[[col, row]] = mat[[row, col]]
        #             new_matrix = IntegerMatrix(mat)
        #             scene.remove(matrix)
        #             scene.play(TransformFromCopy(
        #                 matrix, new_matrix))

        #             matrix = new_matrix
        #             break

        #         row_target.move_to(matrix.get_rows()[row])
        #         scene.play(MoveToTarget(row_rect))

        #     scene.remove(row_rect)
        #     col_rect.move_to(matrix.get_columns()[col+1])
        #     scene.play(TransformFromCopy(row_rect, col_rect))
