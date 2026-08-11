import torch
import torch.nn.functional as F

from model import DistillLoss


class _DirectionalDistillLoss(DistillLoss):
    """One directed cross-view cluster-alignment term for two SimGCD views."""

    def _cross_entropy(self, student_logits, teacher_probabilities):
        return torch.sum(
            -teacher_probabilities * F.log_softmax(student_logits, dim=-1),
            dim=-1,
        ).mean()


class GlobalToForegroundDistillLoss(_DirectionalDistillLoss):
    """Use global view probabilities as teacher for the foreground view."""

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / teacher_temp, dim=-1).detach().chunk(2)
        return self._cross_entropy(student_out[1], teacher_out[0])


class ForegroundToGlobalDistillLoss(_DirectionalDistillLoss):
    """Use foreground view probabilities as teacher for the global view."""

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / teacher_temp, dim=-1).detach().chunk(2)
        return self._cross_entropy(student_out[0], teacher_out[1])
