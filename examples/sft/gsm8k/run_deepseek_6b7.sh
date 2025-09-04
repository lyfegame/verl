# Set project
PROJECT_ID=fundamental-labs

# Create service account
gcloud iam service-accounts create vm-full-access \
    --display-name="VM Full Access Service Account" \
    --project=$PROJECT_ID

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:vm-full-access@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/editor"

# Or more specific roles:
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:vm-full-access@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/compute.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:vm-full-access@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"