from rest_framework import serializers
from testApp.models import ModelResult


class ModelResultSerializer(serializers.Serializer):
    # class Meta:
    #     model = ModelResult
    #     fields = '__all__'
    
    
    #위의 코드 간략화 실행 안됨
    id = serializers.CharField(read_only=True)
    story = serializers.CharField(required=True, max_length=200)
    prediction = serializers.CharField(required=True, max_length=4)

    def create(self, validated_data):
        """
        Create and return a new `ModelTest
        ` instance, given the validated data.
        """
        return ModelResult.objects.create(**validated_data)

    def update(self, instance, validated_data):
        """
        Update and return an existing `ModelTest` instance, given the validated data.
        """
        instance.id = validated_data.get('id', instance.id)
        instance.story = validated_data.get('story', instance.story)
        instance.prediction = validated_data.get('prediction', instance.prediction)
        instance.save()
        return instance