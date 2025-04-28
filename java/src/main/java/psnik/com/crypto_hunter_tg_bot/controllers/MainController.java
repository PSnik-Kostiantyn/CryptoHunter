package psnik.com.crypto_hunter_tg_bot.controllers;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import psnik.com.crypto_hunter_tg_bot.bot.TelegramBot;
import psnik.com.crypto_hunter_tg_bot.entities.EncryptedData;

@RestController
public class MainController {

    private final TelegramBot telegramBot;

    @Autowired
    public MainController(TelegramBot telegramBot) {
        this.telegramBot = telegramBot;
    }

    @PostMapping("/receive")
    public ResponseEntity<EncryptedData> receiveSignals(@RequestBody EncryptedData userData) {
        try {
            EncryptedData responseUserData = new EncryptedData(userData.getIv(), userData.getCt());
            telegramBot.sendUserSignals(userData);
            return ResponseEntity.ok(responseUserData);
        } catch (Exception e){
            return ResponseEntity.internalServerError().body(null);
        }
    }

}
