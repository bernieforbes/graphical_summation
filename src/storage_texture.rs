use anyhow::*;

pub struct StorageTexture {
    #[allow(unused)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub dimensions: wgpu::Extent3d,
}

impl StorageTexture {
    pub fn from_texture(
        diffuse_texture: &wgpu::Texture,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Result<Self> {
        let dimensions = diffuse_texture.size();

        // Storage texture based on the diffuse texture
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: dimensions,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format,
            usage: usage,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            texture,
            view,
            dimensions,
        })
    }
}
