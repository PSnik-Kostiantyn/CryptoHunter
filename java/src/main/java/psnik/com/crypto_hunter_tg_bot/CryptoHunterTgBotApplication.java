package psnik.com.crypto_hunter_tg_bot;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.util.Collections;

@SpringBootApplication
public class CryptoHunterTgBotApplication {

	public static void main(String[] args) {
		SpringApplication app = new SpringApplication(CryptoHunterTgBotApplication.class);
		app.setDefaultProperties(Collections.singletonMap("server.port", getAssignedPort()));
		app.run(args);
		System.out.println("Application has started successfully");
	}

	private static int getAssignedPort() {
		String port = System.getenv("SERVER_PORT");
		if (port != null) {
			return Integer.parseInt(port);
		}
		return 8080;
	}
}
