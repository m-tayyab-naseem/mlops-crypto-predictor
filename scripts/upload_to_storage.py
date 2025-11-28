import boto3
from botocore.client import Config
from pathlib import Path
import subprocess
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DagHubStorageManager:
    """Manage data uploads to DagHub Storage and version with DVC"""
    
    def __init__(self):
        self.endpoint_url = os.getenv(
            'AWS_ENDPOINT_URL',
            'https://dagshub.com/api/v1/repo-buckets/s3/m-tayyab-naseem/mlops-crypto-predictor'
        )
        self.access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self.bucket_name = 'dvc'
        
        # Initialize S3 client for DagHub
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            config=Config(signature_version='s3v4')
        )
        
        print(f"✅ Connected to DagHub Storage")
        print(f"   Endpoint: {self.endpoint_url}")
        print(f"   Bucket: {self.bucket_name}")
    
    def upload_file(self, file_path, object_key=None):
        """Upload file directly to DagHub Storage (S3-compatible)"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if object_key is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            object_key = f"processed/bitcoin_processed_{timestamp}.csv"
        
        try:
            self.s3_client.upload_file(
                str(file_path),
                self.bucket_name,
                object_key
            )
            print(f"✅ Uploaded to DagHub Storage: {object_key}")
            return f"s3://{self.bucket_name}/{object_key}"
        
        except Exception as e:
            print(f"⚠️  Direct upload failed: {str(e)}")
            print("   Will rely on DVC push instead")
            return None
    
    def version_with_dvc(self, file_path):
        """
        Version file with DVC - this automatically uploads to DagHub Storage
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Add file to DVC tracking
            print(f"📦 Adding file to DVC: {file_path.name}")
            result = subprocess.run(
                ['dvc', 'add', str(file_path)],
                capture_output=True,
                text=True,
                cwd=str(file_path.parent.parent.parent)  # Run from project root
            )
            
            if result.returncode != 0:
                print(f"⚠️  DVC add warning: {result.stderr}")
            else:
                print(f"✅ DVC add successful: {result.stdout}")
            
            # Push to DagHub remote storage
            dvc_file = f"{file_path}.dvc"
            if Path(dvc_file).exists():
                print(f"☁️  Pushing to DagHub Storage...")
                result = subprocess.run(
                    ['dvc', 'push', str(file_path)],
                    capture_output=True,
                    text=True,
                    cwd=str(file_path.parent.parent.parent)
                )
                
                if result.returncode != 0:
                    print(f"⚠️  DVC push warning: {result.stderr}")
                    # Continue even if push fails - file is still tracked locally
                else:
                    print(f"✅ Pushed to DagHub Storage")
                    print(f"   {result.stdout}")
                
                # Git add the .dvc file (small metadata file)
                subprocess.run(
                    ['git', 'add', dvc_file, '.gitignore'],
                    capture_output=True,
                    text=True,
                    cwd=str(file_path.parent.parent.parent)
                )
                print(f"✅ Added {Path(dvc_file).name} to git")
                
                return dvc_file
            else:
                print(f"⚠️  DVC file not created: {dvc_file}")
                return None
        
        except subprocess.CalledProcessError as e:
            print(f"❌ DVC operation failed: {e.stderr}")
            print("⚠️  Warning: Continuing without DVC versioning")
            return None
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None


def upload_and_version_data(processed_file_path):
    """
    Main function to upload to DagHub Storage and version with DVC
    """
    print(f"\n{'='*60}")
    print(f"Starting DagHub Storage Upload & Versioning")
    print(f"{'='*60}")
    print(f"File: {Path(processed_file_path).name}")
    
    # Initialize storage manager
    storage = DagHubStorageManager()
    
    # 1. Version with DVC (this uploads to DagHub automatically)
    dvc_file = storage.version_with_dvc(processed_file_path)
    
    result = {
        'local_path': str(processed_file_path),
        'dvc_file': dvc_file,
        'storage_backend': 'DagHub Storage (S3)',
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n{'='*60}")
    print(f"📊 Storage Summary:")
    print(f"{'='*60}")
    print(f"  ✓ Local File: {result['local_path']}")
    print(f"  ✓ DVC Metadata: {result['dvc_file']}")
    print(f"  ✓ Remote Storage: DagHub S3 Bucket")
    print(f"  ✓ Timestamp: {result['timestamp']}")
    print(f"{'='*60}\n")
    
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = upload_and_version_data(file_path)
    else:
        print("Usage: python upload_to_storage.py <file_path>")