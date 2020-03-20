freeze:
	@pip freeze > requirements.txt

convert:
	@tensorflowjs_converter --input_format keras digits_model.h5 ./static/model/