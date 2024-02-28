import numpy as np


class RLEMaskProcessor:  
    @staticmethod
    def rle_encode(img: np.ndarray) -> str:
        '''
        Encode a mask into a run-length encoding.
        
        Args:
            img (numpy.ndarray): Binary mask array where 1 indicates the mask and 0 the background.
        
        Returns:
            str: Run-length encoded string of the mask.
        '''
        pixels = img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)


    @staticmethod
    def rle_decode(mask_rle: str, shape: tuple = (768, 768)) -> np.ndarray:
        '''
        Decode a run-length encoded mask string into a binary mask.
        
        Args:
            mask_rle (str): Run-length encoded mask string.
            shape (tuple): Shape of the output binary mask (height, width).
        
        Returns:
            numpy.ndarray: Binary mask array.
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for start, end in zip(starts, ends):
            img[start:end] = 1
        return img.reshape(shape).T


    @staticmethod
    def masks_as_image(in_mask_list: list) -> np.ndarray:
        '''
        Combine individual run-length encoded masks into a single mask.
        
        Args:
            in_mask_list (list): List of run-length encoded mask strings.
        
        Returns:
            numpy.ndarray: Combined binary mask array.
        '''
        all_masks = np.zeros((768, 768), dtype=np.int16)
        for mask in in_mask_list:
            if isinstance(mask, str):
                all_masks += RLEMaskProcessor.rle_decode(mask)
        
        return np.expand_dims(all_masks, -1)


    @staticmethod
    def multi_rle_encode(img):
        from skimage.morphology import label
        labels = label(img[:, :, 0])
        return [RLEMaskProcessor.rle_encode(labels==k) for k in np.unique(labels[labels>0])]