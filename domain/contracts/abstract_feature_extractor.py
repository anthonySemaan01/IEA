from abc import ABC


class AbstractFeatureExtractor(ABC):

    def get_total_pixels(self, path) -> int:
        pass

    def get_total_blackpixels(self, path) -> int:
        pass

    def get_total_white_pixels(self, path) -> int:
        pass

    def get_total_left_pixels(self, path) -> int:
        pass

    def get_total_right_pixels(self, path) -> int:
        pass

    def get_total_up_pixels(self, path) -> int:
        pass

    def get_total_down_pixels(self, path) -> int:
        pass

    def get_sub_pixels1(self, path) -> int:
        pass

    def get_sub_pixels2(self, path) -> int:
        pass

    def get_sub_pixels3(self, path) -> int:
        pass

    def get_sub_pixels4(self, path) -> int:
        pass

    def get_sub_pixels5(self, path) -> int:
        pass

    def get_sub_pixels6(self, path) -> int:
        pass

    def get_sub_pixels7(self, path) -> int:
        pass

    