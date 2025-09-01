from typing import Optional, List, Literal, Dict, Any
import re
from pydantic import BaseModel, Field, computed_field, validator, ConfigDict, RootModel

class DiskInfo(BaseModel):
    """Information about disk storage allocation.
    
    Attributes:
        total: Total disk space in bytes
        allocated: Currently allocated disk space in bytes
    """
    total: int
    allocated: int

class VMConfig(BaseModel):
    """Configuration for creating a new VM.
    
    Note: Memory and disk sizes should be specified with units (e.g., "4GB", "64GB")
    
    Attributes:
        name: Name of the virtual machine
        os: Operating system type, either "macOS" or "linux"
        cpu: Number of CPU cores to allocate
        memory: Amount of memory to allocate with units
        disk_size: Size of the disk to create with units
        display: Display resolution in format "widthxheight"
        ipsw: IPSW path or 'latest' for macOS VMs, None for other OS types
    """
    name: str
    os: Literal["macOS", "linux"] = "macOS"
    cpu: int = Field(default=2, ge=1)
    memory: str = "4GB"
    disk_size: str = Field(default="64GB", alias="diskSize")
    display: str = "1024x768"
    ipsw: Optional[str] = Field(default=None, description="IPSW path or 'latest', for macOS VMs")

    class Config:
        populate_by_alias = True

class SharedDirectory(BaseModel):
    """Configuration for a shared directory.
    
    Attributes:
        host_path: Path to the directory on the host system
        read_only: Whether the directory should be mounted as read-only
    """
    host_path: str = Field(..., alias="hostPath")  # Allow host_path but serialize as hostPath
    read_only: bool = False
    
    class Config:
        populate_by_name = True  # Allow both alias and original name
        alias_generator = lambda s: ''.join(word.capitalize() if i else word for i, word in enumerate(s.split('_')))

class VMRunOpts(BaseModel):
    """Configuration for running a VM.
    
    Args:
        no_display: Whether to not display the VNC client
        shared_directories: List of directories to share with the VM
    """
    no_display: bool = Field(default=False, alias="noDisplay")
    shared_directories: Optional[list[SharedDirectory]] = Field(
        default=None, 
        alias="sharedDirectories"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=lambda s: ''.join(word.capitalize() if i else word for i, word in enumerate(s.split('_')))
    )

    def model_dump(self, **kwargs):
        """Export model data with proper field name conversion.
        
        Converts shared directory fields to match API expectations when using aliases.
        
        Args:
            **kwargs: Keyword arguments passed to parent model_dump method
            
        Returns:
            dict: Model data with properly formatted field names
        """
        data = super().model_dump(**kwargs)
        # Convert shared directory fields to match API expectations
        if self.shared_directories and "by_alias" in kwargs and kwargs["by_alias"]:
            data["sharedDirectories"] = [
                {
                    "hostPath": d.host_path,
                    "readOnly": d.read_only
                }
                for d in self.shared_directories
            ]
            # Remove the snake_case version if it exists
            data.pop("shared_directories", None)
        return data

class VMStatus(BaseModel):
    """Status information for a virtual machine.
    
    Attributes:
        name: Name of the virtual machine
        status: Current status of the VM
        os: Operating system type
        cpu_count: Number of CPU cores allocated
        memory_size: Amount of memory allocated in bytes
        disk_size: Disk storage information
        vnc_url: URL for VNC connection if available
        ip_address: IP address of the VM if available
    """
    name: str
    status: str
    os: Literal["macOS", "linux"]
    cpu_count: int = Field(alias="cpuCount")
    memory_size: int = Field(alias="memorySize")  # API returns memory size in bytes
    disk_size: DiskInfo = Field(alias="diskSize")
    vnc_url: Optional[str] = Field(default=None, alias="vncUrl")
    ip_address: Optional[str] = Field(default=None, alias="ipAddress")

    class Config:
        populate_by_alias = True

    @computed_field
    @property
    def state(self) -> str:
        """Get the current state of the VM.
        
        Returns:
            str: Current VM status
        """
        return self.status

    @computed_field
    @property
    def cpu(self) -> int:
        """Get the number of CPU cores.
        
        Returns:
            int: Number of CPU cores allocated to the VM
        """
        return self.cpu_count

    @computed_field
    @property
    def memory(self) -> str:
        """Get memory allocation in human-readable format.
        
        Returns:
            str: Memory size formatted as "{size}GB"
        """
        # Convert bytes to GB
        gb = self.memory_size / (1024 * 1024 * 1024)
        return f"{int(gb)}GB"

class VMUpdateOpts(BaseModel):
    """Options for updating VM configuration.
    
    Attributes:
        cpu: Number of CPU cores to update to
        memory: Amount of memory to update to with units
        disk_size: Size of disk to update to with units
    """
    cpu: Optional[int] = None
    memory: Optional[str] = None
    disk_size: Optional[str] = None

class ImageRef(BaseModel):
    """Reference to a VM image.
    
    Attributes:
        image: Name of the image
        tag: Tag version of the image
        registry: Registry hostname where image is stored
        organization: Organization or namespace in the registry
    """
    image: str
    tag: str = "latest"
    registry: Optional[str] = "ghcr.io"
    organization: Optional[str] = "trycua"

    def model_dump(self, **kwargs):
        """Override model_dump to return just the image:tag format.
        
        Args:
            **kwargs: Keyword arguments (ignored)
            
        Returns:
            str: Image reference in "image:tag" format
        """
        return f"{self.image}:{self.tag}"

class CloneSpec(BaseModel):
    """Specification for cloning a VM.
    
    Attributes:
        name: Name of the source VM to clone
        new_name: Name for the new cloned VM
    """
    name: str
    new_name: str = Field(alias="newName")

    class Config:
        populate_by_alias = True

class ImageInfo(BaseModel):
    """Model for individual image information.
    
    Attributes:
        imageId: Unique identifier for the image
    """
    imageId: str

class ImageList(RootModel):
    """Response model for the images endpoint.
    
    A list-like container for ImageInfo objects that provides
    iteration and indexing capabilities.
    """
    root: List[ImageInfo]

    def __iter__(self):
        """Iterate over the image list.
        
        Returns:
            Iterator over ImageInfo objects
        """
        return iter(self.root)

    def __getitem__(self, item):
        """Get an item from the image list by index.
        
        Args:
            item: Index or slice to retrieve
            
        Returns:
            ImageInfo or list of ImageInfo objects
        """
        return self.root[item]

    def __len__(self):
        """Get the number of images in the list.
        
        Returns:
            int: Number of images in the list
        """
        return len(self.root)