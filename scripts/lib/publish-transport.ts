import { createReadStream } from 'node:fs';
import { PutObjectCommand } from '@aws-sdk/client-s3';
import type { S3Client } from '@aws-sdk/client-s3';

interface BlobUpload {
  bucket: string;
  key: string;
  file: string;
  size: number;
  contentType: string;
  sha256: string;
}

// Each call opens a fresh stream so the publisher can retry without buffering files.
export function uploadBlob(s3: S3Client, upload: BlobUpload) {
  return s3.send(
    new PutObjectCommand({
      Bucket: upload.bucket,
      Key: upload.key,
      Body: createReadStream(upload.file),
      ContentLength: upload.size,
      ContentType: upload.contentType,
      ChecksumSHA256: Buffer.from(upload.sha256, 'hex').toString('base64'),
      Metadata: { sha256: upload.sha256 },
      IfNoneMatch: '*',
    }),
  );
}
