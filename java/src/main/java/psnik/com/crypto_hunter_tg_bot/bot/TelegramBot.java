package psnik.com.crypto_hunter_tg_bot.bot;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.telegram.telegrambots.bots.TelegramLongPollingBot;
import org.telegram.telegrambots.meta.api.methods.commands.SetMyCommands;
import org.telegram.telegrambots.meta.api.methods.send.SendMessage;
import org.telegram.telegrambots.meta.api.objects.Update;
import org.telegram.telegrambots.meta.api.objects.commands.BotCommand;
import org.telegram.telegrambots.meta.api.objects.commands.scope.BotCommandScopeDefault;
import org.telegram.telegrambots.meta.api.objects.replykeyboard.InlineKeyboardMarkup;
import org.telegram.telegrambots.meta.api.objects.replykeyboard.buttons.InlineKeyboardButton;
import org.telegram.telegrambots.meta.exceptions.TelegramApiException;
import psnik.com.crypto_hunter_tg_bot.config.BotConfig;
import psnik.com.crypto_hunter_tg_bot.entities.EncryptedData;
import psnik.com.crypto_hunter_tg_bot.service.AesEncryptionService;
import psnik.com.crypto_hunter_tg_bot.service.UserService;

import java.util.*;

@Component
public class TelegramBot extends TelegramLongPollingBot {

    private final BotConfig botConfig;
    private final UserService userService;
    private final AesEncryptionService aesEncryptionService;

    private ResourceBundle messages;

    @Autowired
    public TelegramBot(BotConfig botConfig, UserService userService, AesEncryptionService aesEncryptionService) {
        this.botConfig = botConfig;
        this.userService = userService;
        this.aesEncryptionService = aesEncryptionService;
        setupBotCommands();
    }

    @Override
    public String getBotUsername() {
        return botConfig.getBotName();
    }

    @Override
    public String getBotToken() {
        return botConfig.getToken();
    }

    @Override
    public void onUpdateReceived(Update update) {
        try {
            if (update.hasMessage() && update.getMessage().hasText()) {
                handleTextMessage(update);
            } else if (update.hasCallbackQuery()) {
                handleCallbackQuery(update);
            }
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
        }
    }

    private void handleTextMessage(Update update) {
        long chatId = update.getMessage().getChatId();
        String text = update.getMessage().getText();
        String languageCode = update.getMessage().getFrom().getLanguageCode();
        setMessages(languageCode);

        if (text.equals("/start")) {
            sendWelcomeMessage(chatId);
        } else if (isNumeric(text)) {
            processNumber(chatId, text);
        } else {
            sendMessage(chatId, messages.getString("ask_send_number"));
        }
    }

    private void handleCallbackQuery(Update update) {
        long chatId = update.getCallbackQuery().getMessage().getChatId();
        String languageCode = update.getCallbackQuery().getFrom().getLanguageCode();
        setMessages(languageCode);

        String callbackData = update.getCallbackQuery().getData();
        if ("next_action".equals(callbackData)) {
            sendMessage(chatId, messages.getString("after_button_click"));
        }
    }

    private void processNumber(long chatId, String text) {
        try {
            int number = Integer.parseInt(text);
            ResponseEntity<String> response = userService.postNumberByUserID(String.valueOf(chatId), number);

            if (response.getStatusCode().is2xxSuccessful()) {
                sendMessage(chatId, messages.getString("success_number_received"));
            } else {
                sendMessage(chatId, messages.getString("error_processing_number"));
            }
        } catch (NumberFormatException e) {
            sendMessage(chatId, messages.getString("ask_send_number"));
        } catch (Exception e) {
            sendMessage(chatId, messages.getString("error"));
            System.err.println("Error processing number: " + e.getMessage());
        }
    }

    private void sendWelcomeMessage(long chatId) {
        SendMessage message = new SendMessage();
        message.setChatId(String.valueOf(chatId));
        message.setText(messages.getString("welcome"));
        message.enableHtml(true);

        InlineKeyboardButton button = new InlineKeyboardButton();
        button.setText(messages.getString("press_button"));
        button.setCallbackData("next_action");

        List<List<InlineKeyboardButton>> rows = new ArrayList<>();
        rows.add(Collections.singletonList(button));

        InlineKeyboardMarkup markup = new InlineKeyboardMarkup();
        markup.setKeyboard(rows);

        message.setReplyMarkup(markup);

        try {
            execute(message);
        } catch (TelegramApiException e) {
            System.err.println("Error sending welcome message: " + e.getMessage());
        }
    }

    public void sendMessage(long chatId, String text) {
        try {
            SendMessage message = new SendMessage();
            message.setChatId(String.valueOf(chatId));
            message.setText(text);
            message.enableHtml(true);
            execute(message);
        } catch (TelegramApiException e) {
            System.err.println("Error sending message: " + e.getMessage());
        }
    }

    private void setupBotCommands() {
        List<BotCommand> commands = Collections.singletonList(
                new BotCommand("start", "Start the bot")
        );
        try {
            execute(new SetMyCommands(commands, new BotCommandScopeDefault(), "en"));
            execute(new SetMyCommands(commands, new BotCommandScopeDefault(), "uk"));
            execute(new SetMyCommands(commands, new BotCommandScopeDefault(), "es"));
        } catch (TelegramApiException e) {
            System.err.println("Error setting bot commands: " + e.getMessage());
        }
    }

    private void setMessages(String languageCode) {
        try {
            Locale locale = Locale.forLanguageTag(languageCode);
            messages = ResourceBundle.getBundle("messages", locale);
            if (!messages.getLocale().getLanguage().equals(locale.getLanguage())) {
                messages = ResourceBundle.getBundle("messages", Locale.ENGLISH);
            }
        } catch (MissingResourceException e) {
            messages = ResourceBundle.getBundle("messages", Locale.ENGLISH);
        }
    }

    private boolean isNumeric(String str) {
        try {
            Integer.parseInt(str);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    public void sendUserSignals(EncryptedData userData) throws Exception {
        String decryptedUserData = aesEncryptionService.decryptData(userData.getIv(), userData.getCt());
        System.out.println(decryptedUserData);

        ObjectMapper objectMapper = new ObjectMapper();
        JsonNode rootArrayNode = objectMapper.readTree(decryptedUserData);

        for (JsonNode signal : rootArrayNode) {
            String telegramId = signal.path("telegram_id").asText();
            String ifBelow = signal.path("if_below").asText();
            String messageText = String.format("In the next 24 hours, Bitcoin is expected to fall below %s. 📉 Please note that this is highly uncertain and not financial advice. Good luck! 🚀", ifBelow);
            sendMessage(Long.parseLong(telegramId), messageText);
        }
    }
}
