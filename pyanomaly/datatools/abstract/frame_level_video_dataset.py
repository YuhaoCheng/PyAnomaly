class AvenuePedShanghaiAll(AbstractVideoAnomalyDataset):
    _NAME = 'AvenuePedShanghai'
    def _get_frames(self, video_name):
        cusrsor = self.videos[video_name]['cursor']
        if (cusrsor + self.clip_length) > self.videos[video_name]['length']:
            cusrsor = 0
        if self.mini:
            rng = np.random.RandomState(2020)
            start = rng.randint(0, self.videos[video_name]['length'] - self.clip_length)
        else:
            start = cusrsor

        video_clip, video_clip_original = self.video_loader.read(self.videos[video_name]['frames'], start, start+self.clip_length, clip_length=self.sampled_clip_length, 
                                                                 step=self.frame_step)
        self.videos[video_name]['cursor'] = cusrsor + self.clip_step
        return video_clip
    
    
    def get_image(self, image_name):
        # keep for debug
        image =  self.image_loader.read(image_name)
        return image
