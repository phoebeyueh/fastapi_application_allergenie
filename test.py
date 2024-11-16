def generate_image_from_title(title: str) -> str:
    logger.info(f"Generating image for '{title}'")

    try:
        # Initialize the SageMaker predictor
        predictor = Predictor(endpoint_name=sd_endpoint_name,region_name='us-east-1')
        # Send the payload to the SageMaker endpoint
        prompt = f"professional food photography of {title}, high resolution, appetizing, style plating, restaurant quality"
        response = predictor.predict(prompt, {"ContentType": "application/x-text", "Accept": "application/json"})
        response_json = json.loads(response)
        pil_image = Image.fromarray(np.uint8(response_json["generated_image"]))

        # current_directory = os.getcwd()
        # output_path = os.path.join(current_directory, f"{title}_image.jpg")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'recipe-image-generator'
        formatted_title = quote(title)
        s3_key = f"images/{formatted_title}.jpg"

        # Upload directly from the buffer
        s3.upload_fileobj(buffer, bucket_name, s3_key)
        logger.info(f"Image uploaded directly to S3 bucket '{bucket_name}' with key '{s3_key}'")

        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"

        # Return a clickable link
        return s3_url

    except Exception as e:
        logger.error(f"Error generating image for '{title}': {str(e)}")
        return f"Failed to generate image for '{title}'."
