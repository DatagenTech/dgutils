class ImageHandler:
    '''
    A class to generically handle an image an load it on demand.
    Implements the image lazy loading principle and lets the user load only the required images in the dataset.
    '''
    def __init__(self, path, loading_funcs):
        '''
        Args:
            path: The full path to the image
            loading_funcs: A list of function pointers that will be applied on the image path for loading and pre-processing.
            The functions are applied one after the other, serially. For instance, to read one image and turn it to
            greyscale, one can set it to: [cv2.imread, lambda img : img.mean(axis=-1)]
        '''
        self._path = path
        self._loading_funcs = loading_funcs
        self._img = None

    @property
    def img(self):
        # Load the image if needed
        self.__load()
        return self._img

    def __load(self):
        if self._img is not None:
            return

        # Perform the pre-preocessing with the function pointers
        obj = self._path
        for func in self._loading_funcs:
            obj = func(obj)

        self._img = obj