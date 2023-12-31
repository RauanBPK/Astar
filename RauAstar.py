# A* implementation
# objective -> Minimize f(n)
# WHERE:
# f(n) = g(n) + h(n)
# n = node
# g(n) = cost of the path from the starting node
# h(n) = heuristic function. Estimates lowest cost from n to the goal node

# "If the heuristic function is admissible – meaning that it never overestimates
# the actual cost to get to the goal – A* is guaranteed to return a least-cost path from start to goal."

# h(n) could be defined as:
# for p1 = (p1x, p1y), p2 = (p2x, p2y)
# Euclidean distance -> √((p2x - p1x)^2 + (p2y - p1y)^2) (shortest path "possible" because it is a straight line) (any)
# Manhattan distance -> |p2x - p1x| + |p2y - p1y| (sum of the absolute distances of its coordinates) (no diagonals)
# Chebyshev distance -> max(|p2x - p1x|, |p2y - p1y|) (greatest distance(abs difference) of all coordinates) (diagonals)

# Used this as an example: https://www.youtube.com/watch?v=JtiK0DOeI4A

# I overengineered it a little bit, but irisuariris.

import heapq
import time

import pygame
from typing import List
from utils import Colors
from enum import Enum, auto
from functools import partial


class NodeType(Enum):
    BLANK = auto()
    START = auto()
    END = auto()
    PATH = auto()
    OPEN = auto()
    CLOSED = auto()
    OBSTACLE = auto()


class Node:
    def __init__(self, row, col, node_type: NodeType = NodeType.BLANK):
        self.node_type: NodeType = node_type
        self.row: int = row
        self.col: int = col
        self.parent: Node | None = None
        self.neighbors: List["Node"] = []
        self.g = float("inf")
        self.h = 0

    def change_type(self, new_type: NodeType):
        self.node_type = new_type

    def reset(self, keep_type=False):
        if not keep_type:
            self.node_type = NodeType.BLANK
        self.g = float("inf")
        self.h = 0
        self.parent = None

    def set_g(self, new_g):
        self.g = new_g

    @property
    def f(self):
        return self.g + self.h

    def calculate_heuristic(self, node: "Node", method=None):
        x1, y1 = self.row, self.col
        x2, y2 = node.row, node.col
        if method == "manhattan":
            self.h = abs(x1 - x2) + abs(y1 - y2)
        elif method == "chebyshev":
            self.h = max(abs(x2 - x1), abs(y2 - y1))
        else:  # euclidean
            self.h = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return self.h


class Cell:
    type_colors = {
        NodeType.BLANK: Colors.L_GRAY,
        NodeType.START: Colors.ORANGE,
        NodeType.END: Colors.TURQUOISE,
        NodeType.PATH: Colors.YELLOW,
        NodeType.OPEN: Colors.D_GREEN,
        NodeType.CLOSED: Colors.RED,
        NodeType.OBSTACLE: Colors.BLACK,
    }

    def __init__(self, node: Node, width):
        self.width = width
        self.node = node
        self.color = self.type_colors[node.node_type]
        self.x = self.node.row * self.width
        self.y = self.node.col * self.width

    def change_type(self, new_type: NodeType):
        self.node.change_type(new_type)
        self.color = self.type_colors[new_type]

    def reset(self, keep_type=False):
        self.node.reset(keep_type)
        self.color = self.type_colors[self.node.node_type]

    @property
    def get_color(self):
        return self.type_colors[self.node.node_type]

    def draw_cell(self, window):
        pygame.draw.rect(
            window, self.get_color, (self.x, self.y, self.width, self.width)
        )


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid: List[List[Node]] = []

    def make_grid(self):
        for i in range(self.rows):
            self.grid.append([])
            for j in range(self.cols):
                node = Node(i, j)
                self.grid[i].append(node)

    def cost_to_neighbor(self):
        # This would/could be the function that calculates the cost of moving from one node to an adjacent neighbor.
        # But, since we are on a grid, moving to any imediate neighbor always costs 1 :)
        return 1

    def update_all_neighbors(self, diagonals=False):
        for i in range(self.rows):
            for j in range(self.cols):
                node = self.grid[i][j]
                self.update_node_neighbors(node, diagonals)

    def update_node_neighbors(self, node: Node, diagonals=False):
        node.neighbors = []
        # Top
        if (
            node.row > 0
            and self.grid[node.row - 1][node.col].node_type != NodeType.OBSTACLE
        ):
            node.neighbors.append(self.grid[node.row - 1][node.col])
        # Down
        if (
            node.row < self.rows - 1
            and self.grid[node.row + 1][node.col].node_type != NodeType.OBSTACLE
        ):
            node.neighbors.append(self.grid[node.row + 1][node.col])
        # Left
        if (
            node.col > 0
            and self.grid[node.row][node.col - 1].node_type != NodeType.OBSTACLE
        ):
            node.neighbors.append(self.grid[node.row][node.col - 1])
        # Right
        if (
            node.col < self.rows - 1
            and self.grid[node.row][node.col + 1].node_type != NodeType.OBSTACLE
        ):
            node.neighbors.append(self.grid[node.row][node.col + 1])

        if diagonals:
            # top-right
            if (
                node.col < self.rows - 1
                and node.row > 0
                and self.grid[node.row - 1][node.col + 1].node_type != NodeType.OBSTACLE
            ):  # noqa: E501
                node.neighbors.append(self.grid[node.row - 1][node.col + 1])
            # top-left
            if (
                node.col > 0
                and node.row > 0
                and self.grid[node.row - 1][node.col - 1].node_type != NodeType.OBSTACLE
            ):
                node.neighbors.append(self.grid[node.row - 1][node.col - 1])
            # bottom-right
            if (
                node.row < self.rows - 1
                and node.col < self.rows - 1
                and self.grid[node.row + 1][node.col + 1].node_type != NodeType.OBSTACLE
            ):  # noqa: E501
                node.neighbors.append(self.grid[node.row + 1][node.col + 1])
            # bottom-left
            if (
                node.col > 0
                and node.row < self.rows - 1
                and self.grid[node.row + 1][node.col - 1].node_type != NodeType.OBSTACLE
            ):  # noqa: E501
                node.neighbors.append(self.grid[node.row + 1][node.col - 1])


class GameGrid:
    # We`ll be assuming a square grid with square cells because I said so
    def __init__(self, rows, window_size, window):
        self.rows = rows
        self.logic_grid: Grid = Grid(self.rows, self.rows)
        self.cell_grid: List[List[Cell]] = []
        self.window_size = window_size
        self.window = window
        self.cell_width = window_size // self.rows
        self.draw_info = None

    def get_clicked_cell(self, click_position):
        y, x = click_position
        row = y // self.cell_width
        col = x // self.cell_width
        clicked_cell = self.cell_grid[row][col]
        return clicked_cell

    def draw_grid_lines(self):
        for i in range(self.rows):
            pygame.draw.line(
                self.window,
                Colors.GREY,
                (0, i * self.cell_width),
                (self.window_size, i * self.cell_width),
            )
            pygame.draw.line(
                self.window,
                Colors.GREY,
                (i * self.cell_width, 0),
                (i * self.cell_width, self.window_size),
            )

    def make_cell_grid(self):
        self.logic_grid.make_grid()
        for i, row in enumerate(self.logic_grid.grid):
            self.cell_grid.append([])
            for j, node in enumerate(row):
                cell = Cell(node, self.cell_width)
                self.cell_grid[i].append(cell)

    def reset(self, clear_user_input=True):
        if clear_user_input:
            for i in range(self.rows):
                for j in range(self.rows):
                    self.cell_grid[i][j].reset()
        else:
            for i in range(self.rows):
                for j in range(self.rows):
                    if self.cell_grid[i][j].node.node_type not in [
                        NodeType.START,
                        NodeType.END,
                        NodeType.OBSTACLE,
                    ]:
                        self.cell_grid[i][j].reset()
                    else:
                        self.cell_grid[i][j].reset(keep_type=True)

    def set_draw_info(self, func, **params):
        # partial is fun. Sorry to whoever is reading this (unless you are me, then I am not so sorry)
        self.draw_info = partial(func, **params) if func else None

    def draw(self):
        self.window.fill(Colors.WHITE)
        for row in self.cell_grid:
            for cell in row:
                cell.draw_cell(self.window)
        self.draw_grid_lines()
        if self.draw_info:
            self.draw_info()
        pygame.display.update()


class MyHeap:
    def __init__(self):
        self.heap = []

    def reset(self):
        self.heap = []

    def is_empty(self):
        return len(self.heap) == 0

    def push(self, item):
        heapq.heappush(self.heap, item)

    def pop(self):
        item = heapq.heappop(self.heap)
        return item

    def contains(self, item):
        return any([item in obj for obj in self.heap])


class AstarAlgorithm:
    def __init__(
        self, grid: Grid, start: Node, end: Node, diagonals=False, heuristic=None
    ):
        self.start = start
        self.end = end
        self.grid = grid
        self.open_set = MyHeap()
        self.diagonals = diagonals
        self.heuristic = heuristic
        self.draw_callback = None  # Callback to be called after each step
        self.check_event_callback = None

    def setup(self):
        self.grid.update_all_neighbors(diagonals=self.diagonals)
        self.start.set_g(0)
        self.start.calculate_heuristic(self.end, method=self.heuristic)
        self.open_set.reset()

    def reconstruct_path(self, from_node: Node):
        current = from_node
        path = []
        while current != self.start:
            current = current.parent
            path.append(current)
        path.remove(current)
        for node in reversed(path):
            node.change_type(NodeType.PATH)
            if self.draw_callback:
                self.draw_callback()
                time.sleep(0.05)  # Just to better see the path forming

    def run(self):
        self.setup()
        count = 0
        # start by adding the start node to the open set
        self.open_set.push((self.start.f, count, self.start))  # not sure about this

        while not self.open_set.is_empty():
            if self.check_event_callback:
                self.check_event_callback()
            current_node: Node = self.open_set.pop()[2]

            if current_node == self.end:
                done = time.time()
                self.reconstruct_path(current_node)
                return done

            for neighbor in current_node.neighbors:
                temp_g_score = current_node.g + self.grid.cost_to_neighbor()

                if (
                    temp_g_score < neighbor.g
                ):  # if this is currently the best way to get to "neighbor"
                    neighbor.parent = current_node
                    neighbor.set_g(temp_g_score)
                    neighbor.calculate_heuristic(self.end, method=self.heuristic)
                    if not self.open_set.contains(neighbor):
                        count += 1
                        self.open_set.push((neighbor.f, count, neighbor))
                        if neighbor != self.end:
                            neighbor.change_type(NodeType.OPEN)

            if self.draw_callback:
                self.draw_callback()

            if current_node != self.start:
                current_node.change_type(NodeType.CLOSED)

        done = time.time()
        return done


class AstarVisualization:
    def __init__(
        self,
        game_grid: GameGrid,
        start_cell: Cell,
        end_cell: Cell,
        diagonals=False,
        heuristic=None,
    ):
        self.game_grid: GameGrid = game_grid
        self.start_cell = start_cell
        self.end_cell = end_cell
        self.diagonals = diagonals
        self.heuristic = heuristic
        self.astar = AstarAlgorithm(
            self.game_grid.logic_grid,
            self.start_cell.node,
            self.end_cell.node,
            self.diagonals,
            self.heuristic,
        )  # noqa

    def visualize_step(self):
        self.game_grid.draw()

    def check_event_callback(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def run(self):
        self.astar.draw_callback = self.visualize_step
        self.astar.check_event_callback = self.check_event_callback
        return self.astar.run()


class Game:
    def __init__(self, window, window_size, rows, diagonals=False, heuristics=None):
        self.window = window
        self.window_size = window_size
        self.rows = rows
        self.diagonals = diagonals
        self.heuristics = heuristics
        self.total_time = 0

        self.astar = None
        self.start = None
        self.end = None
        self.running = True
        self.game_grid = GameGrid(rows, window_size, window)

    def draw_info(self, diagonals, heuristic, total_time):
        info_text = f"Diagonals: {diagonals}, Heuristics: {str.capitalize(heuristic)}, Execution time: {total_time:.3f}s"
        font = pygame.font.Font(
            pygame.font.get_default_font(),
            ((self.window_size * 2) - 10) // len(info_text),
        )
        text_surface = font.render(info_text, True, Colors.BLACK)
        self.window.blit(text_surface, (10, 10))
        pygame.display.update()

    def update_info(self):
        if self.game_grid.draw_info:
            self.game_grid.set_draw_info(
                self.draw_info,
                diagonals=self.diagonals,
                heuristic=self.heuristics,
                total_time=self.total_time,
            )

    def setup(self):
        self.game_grid.make_cell_grid()
        self.game_grid.set_draw_info(
            self.draw_info,
            diagonals=self.diagonals,
            heuristic=self.heuristics,
            total_time=self.total_time,
        )

    def run(self):
        self.setup()
        while self.running:
            self.game_grid.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if pygame.mouse.get_pressed()[0]:  # left click
                    click_pos = pygame.mouse.get_pos()
                    clicked_cell = self.game_grid.get_clicked_cell(click_pos)
                    if not self.start and clicked_cell != self.end:
                        self.start = clicked_cell
                        self.start.change_type(NodeType.START)
                    elif not self.end and clicked_cell != self.start:
                        self.end = clicked_cell
                        self.end.change_type(NodeType.END)
                    elif clicked_cell not in [self.start, self.end]:
                        clicked_cell.change_type(NodeType.OBSTACLE)

                elif pygame.mouse.get_pressed()[2]:  # right click
                    click_pos = pygame.mouse.get_pos()
                    clicked_cell = self.game_grid.get_clicked_cell(click_pos)
                    clicked_cell.reset()
                    if clicked_cell == self.start:
                        self.start = None
                    elif clicked_cell == self.end:
                        self.end = None

                # start A*
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.start and self.end:
                        self.astar = AstarVisualization(
                            self.game_grid,
                            self.start,
                            self.end,
                            self.diagonals,
                            self.heuristics,
                        )
                        self.game_grid.reset(clear_user_input=False)
                        time_before = time.time()
                        done = self.astar.run()
                        self.total_time = done - time_before
                        self.update_info()
                        print(f"Execution took {self.total_time:.3f}s")
                    if event.key == pygame.K_c:  # Reset grid
                        self.start = None
                        self.end = None
                        self.game_grid.reset()
                    if event.key == pygame.K_h:  # Change heuristic
                        hs = ["manhattan", "chebyshev", "euclidean"]
                        current_h_index = hs.index(self.heuristics)
                        try:
                            next_h = hs[current_h_index + 1]
                        except IndexError:
                            next_h = hs[0]
                        self.heuristics = next_h
                        self.update_info()
                        print(f"Heuristic updated to: {self.heuristics}")
                    if event.key == pygame.K_d:  # Toggle diagonals
                        self.diagonals = not self.diagonals
                        self.update_info()
                        print(f"Diagonals updated to: {self.diagonals}")
                    if event.key == pygame.K_i:
                        if self.game_grid:
                            # injects a function (like a callback?) with the parameters already set
                            # just so the drawing is done in one place only. If I draw it here the info would
                            # disappear while the algorithm is running... Maybe I just don`t know how to do things
                            # good ol friend tunnel vison
                            self.game_grid.set_draw_info(
                                self.draw_info,
                                diagonals=self.diagonals,
                                heuristic=self.heuristics,
                                total_time=self.total_time,
                            ) if not self.game_grid.draw_info else self.game_grid.set_draw_info(
                                None
                            )

        pygame.quit()


if __name__ == "__main__":
    WINDOW_WIDTH = 800
    WIN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_WIDTH))
    pygame.display.set_caption("A* Algorithm Playground")
    ROWS = 40
    DIAG = False
    HEURISTIC = "manhattan"  # "manhattan", "chebyshev", "euclidean"
    new_game = Game(WIN, WIN.get_width(), ROWS, diagonals=DIAG, heuristics=HEURISTIC)
    pygame.font.init()
    new_game.run()
