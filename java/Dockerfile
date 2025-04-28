FROM eclipse-temurin:17-jdk-alpine

RUN apk add --no-cache maven

WORKDIR /app

COPY . .

RUN mvn clean package

EXPOSE 8001

ENTRYPOINT ["java", "-jar", "target/crypto_hunter_tg_bot-0.0.1-SNAPSHOT.jar"]
