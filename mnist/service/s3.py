import boto3


class S3:
    """
    Service class handling connection with AWS S3
    """

    def __init__(self, region_name, access_key, secret_key, bucket):
        self.region_name = region_name
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket

    def _boto3_client(self):
        """
        creates a boto3 client instance
        """
        boto3_client = boto3.client(
            "s3",
            region_name=self.region_name,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        return boto3_client

    def upload(self, filename: str, key: str):
        """
        Uploads file to S3

        filename: The path to the file to upload.
        key: The name of the key to upload to.
        """
        boto3_client = self._boto3_client()
        boto3_client.upload_file(filename, self.bucket, key)
